"""
Machine Learning Forecaster for Epidemic Trajectories & Hospital Burden
Implements XGBoost / Random Forest time-series forecasting with grouped rolling epidemiological features,
multi-horizon projections, and confidence interval estimation.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb

MODELS_DIR = os.path.dirname(os.path.abspath(__file__))

class EpidemicForecaster:
    def __init__(self, model_type="xgboost"):
        self.model_type = model_type
        if model_type == "xgboost":
            self.model = xgb.XGBRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.03,
                subsample=0.85,
                colsample_bytree=0.85,
                random_state=42,
                n_jobs=-1
            )
        else:
            self.model = RandomForestRegressor(
                n_estimators=180,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
        self.feature_names = []
        self.residual_std = 1.0
        self.is_fitted = False

    def create_features(self, df):
        """
        Engineers temporal and epidemiological indicators grouped by region & pathogen:
        - Lags (1, 2, 3, 7, 14 days)
        - Rolling means and volatility (7, 14 days)
        - Growth velocity ratios
        - Testing & Positivity signals
        - Policy and Mobility metrics
        - Day-of-week seasonality
        """
        data = df.copy().sort_values(["region", "pathogen", "date"]).reset_index(drop=True)
        
        data["date_dt"] = pd.to_datetime(data["date"])
        data["day_of_week"] = data["date_dt"].dt.dayofweek
        data["is_weekend"] = data["day_of_week"].isin([5, 6]).astype(int)

        # Compute group-specific lags and rolling features
        group_cols = ["region", "pathogen"] if "region" in data.columns and "pathogen" in data.columns else None

        if group_cols:
            grouped = data.groupby(group_cols)
            for lag in [1, 2, 3, 7, 14]:
                data[f"cases_lag_{lag}"] = grouped["daily_cases"].shift(lag)
            
            data["rolling_mean_7"] = grouped["daily_cases"].shift(1).transform(lambda s: s.rolling(7).mean())
            data["rolling_std_7"] = grouped["daily_cases"].shift(1).transform(lambda s: s.rolling(7).std().fillna(0))
            data["rolling_mean_14"] = grouped["daily_cases"].shift(1).transform(lambda s: s.rolling(14).mean())
            
            data["tpr_lag_1"] = grouped["tpr_pct"].shift(1)
            data["tpr_lag_7"] = grouped["tpr_pct"].shift(7)
        else:
            for lag in [1, 2, 3, 7, 14]:
                data[f"cases_lag_{lag}"] = data["daily_cases"].shift(lag)
            
            data["rolling_mean_7"] = data["daily_cases"].shift(1).rolling(7).mean()
            data["rolling_std_7"] = data["daily_cases"].shift(1).rolling(7).std().fillna(0)
            data["rolling_mean_14"] = data["daily_cases"].shift(1).rolling(14).mean()
            
            data["tpr_lag_1"] = data["tpr_pct"].shift(1)
            data["tpr_lag_7"] = data["tpr_pct"].shift(7)

        # Growth velocity ratio
        data["growth_ratio_7_14"] = (data["rolling_mean_7"] + 1) / (data["rolling_mean_14"] + 1)

        features = [
            "cases_lag_1", "cases_lag_2", "cases_lag_3", "cases_lag_7", "cases_lag_14",
            "rolling_mean_7", "rolling_std_7", "rolling_mean_14", "growth_ratio_7_14",
            "tpr_lag_1", "tpr_lag_7", "mobility_index", "stringency_index",
            "vaccination_coverage_pct", "temperature_c", "humidity_pct",
            "day_of_week", "is_weekend"
        ]

        self.feature_names = features
        return data

    def train(self, df):
        """
        Trains forecaster using chronological train/validation split across groups.
        """
        fe_df = self.create_features(df)
        clean_df = fe_df.dropna(subset=self.feature_names + ["daily_cases"]).reset_index(drop=True)

        X = clean_df[self.feature_names]
        y = clean_df["daily_cases"]

        # Stratified chronological split per group or overall chronological split
        split_idx = int(len(clean_df) * 0.85)
        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

        self.model.fit(X_train, y_train)
        self.is_fitted = True

        val_preds = np.maximum(0, self.model.predict(X_val))
        residuals = y_val - val_preds
        self.residual_std = float(np.std(residuals))

        mae = mean_absolute_error(y_val, val_preds)
        rmse = float(np.sqrt(mean_squared_error(y_val, val_preds)))
        r2 = r2_score(y_val, val_preds)
        mape = float(np.mean(np.abs(residuals / np.maximum(y_val, 1)))) * 100.0

        metrics = {
            "model_type": self.model_type,
            "validation_mae": round(mae, 2),
            "validation_rmse": round(rmse, 2),
            "validation_r2": round(r2, 4),
            "validation_mape_pct": round(mape, 2)
        }
        return metrics

    def forecast(self, historical_df, horizon_days=14):
        """
        Generates multi-step ahead recursive forecasting with 95% confidence bounds.
        """
        if not self.is_fitted:
            raise ValueError("Forecaster model has not been trained yet.")

        fe_df = self.create_features(historical_df)
        current_data = fe_df.copy()

        last_date = pd.to_datetime(current_data["date"].iloc[-1])
        future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, horizon_days + 1)]

        predictions = []
        lower_bounds = []
        upper_bounds = []

        last_row = current_data.iloc[-1]
        cur_mobility = float(last_row.get("mobility_index", 75))
        cur_stringency = float(last_row.get("stringency_index", 40))
        cur_vax = float(last_row.get("vaccination_coverage_pct", 65))
        cur_temp = float(last_row.get("temperature_c", 26))
        cur_humid = float(last_row.get("humidity_pct", 70))
        cur_tpr = float(last_row.get("tpr_pct", 5.0))

        simulated_cases = list(current_data["daily_cases"].values)
        simulated_tpr = list(current_data["tpr_pct"].values)

        for step, f_date in enumerate(future_dates, 1):
            day_of_week = f_date.dayofweek
            is_weekend = 1 if day_of_week in [5, 6] else 0

            c_lag1 = simulated_cases[-1]
            c_lag2 = simulated_cases[-2] if len(simulated_cases) >= 2 else c_lag1
            c_lag3 = simulated_cases[-3] if len(simulated_cases) >= 3 else c_lag1
            c_lag7 = simulated_cases[-7] if len(simulated_cases) >= 7 else c_lag1
            c_lag14 = simulated_cases[-14] if len(simulated_cases) >= 14 else c_lag7

            rmean7 = np.mean(simulated_cases[-7:])
            rstd7 = np.std(simulated_cases[-7:])
            rmean14 = np.mean(simulated_cases[-14:])
            growth_ratio = (rmean7 + 1) / (rmean14 + 1)

            tpr_lag1 = simulated_tpr[-1]
            tpr_lag7 = simulated_tpr[-7] if len(simulated_tpr) >= 7 else tpr_lag1

            feature_dict = {
                "cases_lag_1": [c_lag1],
                "cases_lag_2": [c_lag2],
                "cases_lag_3": [c_lag3],
                "cases_lag_7": [c_lag7],
                "cases_lag_14": [c_lag14],
                "rolling_mean_7": [rmean7],
                "rolling_std_7": [rstd7],
                "rolling_mean_14": [rmean14],
                "growth_ratio_7_14": [growth_ratio],
                "tpr_lag_1": [tpr_lag1],
                "tpr_lag_7": [tpr_lag7],
                "mobility_index": [cur_mobility],
                "stringency_index": [cur_stringency],
                "vaccination_coverage_pct": [cur_vax],
                "temperature_c": [cur_temp],
                "humidity_pct": [cur_humid],
                "day_of_week": [day_of_week],
                "is_weekend": [is_weekend]
            }

            feat_df = pd.DataFrame(feature_dict)[self.feature_names]
            pred = float(np.maximum(0, self.model.predict(feat_df)[0]))

            margin = 1.96 * self.residual_std * np.sqrt(step * 0.35)
            lower = max(0, int(round(pred - margin)))
            upper = int(round(pred + margin))

            predictions.append(int(round(pred)))
            lower_bounds.append(lower)
            upper_bounds.append(upper)

            simulated_cases.append(pred)
            new_tpr = max(0.5, tpr_lag1 * (pred / max(c_lag1, 1)))
            simulated_tpr.append(new_tpr)

        return {
            "dates": [d.strftime("%Y-%m-%d") for d in future_dates],
            "predictions": predictions,
            "lower_95_ci": lower_bounds,
            "upper_95_ci": upper_bounds,
            "horizon_days": horizon_days
        }

    def save(self, file_path=None):
        if file_path is None:
            file_path = os.path.join(MODELS_DIR, f"{self.model_type}_forecaster.joblib")
        joblib.dump({
            "model": self.model,
            "model_type": self.model_type,
            "feature_names": self.feature_names,
            "residual_std": self.residual_std,
            "is_fitted": self.is_fitted
        }, file_path)
        return file_path

    @classmethod
    def load(cls, file_path):
        data = joblib.load(file_path)
        instance = cls(model_type=data["model_type"])
        instance.model = data["model"]
        instance.feature_names = data["feature_names"]
        instance.residual_std = data["residual_std"]
        instance.is_fitted = data["is_fitted"]
        return instance
