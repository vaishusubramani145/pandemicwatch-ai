"""
Epidemiological Risk Assessment Engine
Computes:
- Real-time Effective Reproduction Number (Rt)
- Composite Outbreak Early Warning Index (EWI)
- Healthcare Capacity Stress & Days-to-Exhaustion
- Syndromic Anomaly / Spike Detector
"""

import numpy as np
import pandas as pd

class EpidemicRiskEngine:
    def __init__(self, serial_interval_days=5.0):
        self.serial_interval = serial_interval_days

    def estimate_rt(self, daily_cases, window=7):
        """
        Estimates real-time Rt using the causal rolling window growth rate:
        r = ln(Case_t / Case_{t-window}) / window
        Rt = exp(r * serial_interval)
        """
        cases = np.array(daily_cases, dtype=float)
        n = len(cases)
        if n < window + 1:
            return [1.0] * n

        # Use causal rolling average to prevent boundary zero-padding distortion
        s = pd.Series(cases)
        smoothed = s.rolling(window, min_periods=1).mean().values
        smoothed = np.maximum(smoothed, 1.0)

        rt_series = []
        for i in range(n):
            if i < window:
                rt_series.append(1.0)
            else:
                prev = smoothed[i - window]
                curr = smoothed[i]
                growth_ratio = max(curr / max(prev, 1.0), 0.001)
                r = np.log(growth_ratio) / window
                rt_val = np.exp(r * self.serial_interval)
                rt_series.append(float(np.clip(rt_val, 0.1, 5.0)))

        return [round(x, 2) for x in rt_series]

    def compute_early_warning_index(
        self,
        current_rt,
        weekly_case_growth_pct,
        test_positivity_rate,
        icu_occupancy_pct,
        anomaly_flag=False
    ):
        """
        Calculates Composite Early Warning Index (0 - 100).
        """
        rt_norm = np.clip((current_rt - 0.7) / (2.0 - 0.7), 0.0, 1.0) * 100.0
        growth_norm = np.clip((weekly_case_growth_pct + 20.0) / 120.0, 0.0, 1.0) * 100.0
        tpr_norm = np.clip(test_positivity_rate / 15.0, 0.0, 1.0) * 100.0
        icu_norm = np.clip((icu_occupancy_pct - 20.0) / 75.0, 0.0, 1.0) * 100.0
        anomaly_bonus = 10.0 if anomaly_flag else 0.0

        raw_index = (
            0.30 * rt_norm +
            0.25 * growth_norm +
            0.20 * tpr_norm +
            0.20 * icu_norm +
            0.05 * anomaly_bonus
        )
        ewi = float(np.clip(raw_index, 0.0, 100.0))

        if ewi < 30.0:
            tier = "LOW"
            color = "#10B981"
            action = "Routine Surveillance: Community spread is contained. Maintain standard testing."
        elif ewi < 55.0:
            tier = "MODERATE"
            color = "#F59E0B"
            action = "Enhanced Alert: Emerging local transmission clusters. Increase contact tracing and testing density."
        elif ewi < 75.0:
            tier = "HIGH"
            color = "#F97316"
            action = "High Vigilance: Rapid community propagation. Activate hospital surge beds, recommend indoor masking."
        else:
            tier = "CRITICAL"
            color = "#EF4444"
            action = "Emergency Outbreak: Severe exponential growth risking hospital collapse. Implement targeted gathering caps and triage."

        return {
            "score": round(ewi, 1),
            "tier": tier,
            "color": color,
            "recommended_action": action,
            "components": {
                "rt_score": round(rt_norm, 1),
                "growth_score": round(growth_norm, 1),
                "tpr_score": round(tpr_norm, 1),
                "icu_score": round(icu_norm, 1)
            }
        }

    def detect_outbreak_anomalies(self, daily_cases, z_threshold=2.2):
        cases = np.array(daily_cases, dtype=float)
        n = len(cases)
        anomalies = [False] * n
        if n < 14:
            return anomalies

        window = 10
        for i in range(window, n):
            history = cases[i-window:i]
            mean = np.mean(history)
            std = np.std(history)
            if std > 0.1:
                z_score = (cases[i] - mean) / std
                if z_score >= z_threshold and cases[i] > mean * 1.3:
                    anomalies[i] = True

        return anomalies

    def calculate_hospital_exhaustion(
        self,
        total_icu_beds,
        current_occupied_icu,
        daily_new_icu_admissions,
        mean_los_days=10.0
    ):
        """
        Calculates days until ICU capacity is fully saturated (100%),
        taking current occupancy percentage and net admission rates into account.
        """
        available_beds = total_icu_beds - current_occupied_icu
        occupancy_pct = (current_occupied_icu / total_icu_beds) * 100.0

        if available_beds <= 0:
            return {
                "days_to_exhaustion": 0,
                "current_occupancy_pct": 100.0,
                "status": "EXHAUSTED",
                "risk_level": "CRITICAL"
            }

        daily_discharges = current_occupied_icu / mean_los_days
        net_daily_gain = daily_new_icu_admissions - daily_discharges

        if net_daily_gain <= 0:
            if occupancy_pct >= 90.0:
                status = "WARNING_SURGE"
                risk = "SEVERE"
                days = "At Peak Capacity"
            elif occupancy_pct >= 75.0:
                status = "ELEVATED_STRESS"
                risk = "ELEVATED"
                days = "Plateau Near Peak"
            else:
                status = "STABLE"
                risk = "LOW"
                days = "No Deficit Expected"
        else:
            days = int(np.ceil(available_beds / net_daily_gain))
            if days <= 7 or occupancy_pct >= 85.0:
                status = "CRITICAL_SURGE"
                risk = "SEVERE"
            elif days <= 14 or occupancy_pct >= 75.0:
                status = "WARNING_SURGE"
                risk = "ELEVATED"
            else:
                status = "MANAGEABLE_GROWTH"
                risk = "MODERATE"

        return {
            "days_to_exhaustion": days,
            "current_occupancy_pct": round(occupancy_pct, 1),
            "net_daily_change": round(net_daily_gain, 1),
            "status": status,
            "risk_level": risk
        }
