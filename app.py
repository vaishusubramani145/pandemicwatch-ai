"""
PandemicWatch AI - Flask Application & Epidemiological Intelligence Command Center
Serves REST APIs for ML outbreak forecasting, SEIR policy simulation,
dynamic Rt estimation, hospital capacity strain, and automated health briefings.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify

# Add directory to path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from data.generator import (
    get_or_create_benchmark_dataset,
    generate_epidemiological_timeseries,
    REGIONS,
    PATHOGENS
)
from models.seir_model import SEIRModel
from models.risk_engine import EpidemicRiskEngine
from models.ml_forecaster import EpidemicForecaster

app = Flask(__name__)
app.config['SECRET_KEY'] = 'pandemicwatch-ai-secure-internship-key-2026'

# Initialize risk engine and load cached models
risk_engine = EpidemicRiskEngine()

# Model loading with fallback training if needed
forecaster = None
model_path = os.path.join(BASE_DIR, "models", "xgboost_forecaster.joblib")

def get_forecaster():
    global forecaster
    if forecaster is not None:
        return forecaster
    if os.path.exists(model_path):
        try:
            forecaster = EpidemicForecaster.load(model_path)
            return forecaster
        except Exception as e:
            print(f"Error loading model from {model_path}: {e}")
    
    # Train on the fly if needed
    print("Training forecaster on the fly...")
    df = get_or_create_benchmark_dataset()
    forecaster = EpidemicForecaster(model_type="xgboost")
    forecaster.train(df)
    try:
        forecaster.save(model_path)
    except Exception:
        pass
    return forecaster

@app.route("/")
def index():
    return render_template(
        "index.html",
        regions=list(REGIONS.keys()),
        pathogens=list(PATHOGENS.keys()),
        default_region="India",
        default_pathogen="COVID-19 (SARS-CoV-2)"
    )

@app.route("/api/overview")
def api_overview():
    region = request.args.get("region", "India")
    pathogen = request.args.get("pathogen", "COVID-19 (SARS-CoV-2)")
    
    if region not in REGIONS:
        region = "India"
    if pathogen not in PATHOGENS:
        pathogen = "COVID-19 (SARS-CoV-2)"

    df = get_or_create_benchmark_dataset()
    sub_df = df[(df["region"] == region) & (df["pathogen"] == pathogen)].sort_values("date").reset_index(drop=True)
    if sub_df.empty:
        sub_df = generate_epidemiological_timeseries(region=region, pathogen=pathogen)

    reg_meta = REGIONS[region]
    path_meta = PATHOGENS[pathogen]

    cases = sub_df["daily_cases"].tolist()
    dates = sub_df["date"].tolist()

    # Dynamic Rt calculation
    rt_series = risk_engine.estimate_rt(cases)
    current_rt = rt_series[-1]

    # Weekly case growth rate
    recent_7_sum = sum(cases[-7:])
    prev_7_sum = sum(cases[-14:-7]) if len(cases) >= 14 else recent_7_sum
    growth_pct = ((recent_7_sum - prev_7_sum) / max(prev_7_sum, 1)) * 100.0

    # Test Positivity Rate
    latest_tpr = float(sub_df["tpr_pct"].iloc[-1])

    # ICU occupancy strain
    latest_icu = int(sub_df["icu_active"].iloc[-1])
    total_icu = reg_meta["icu_beds"]
    icu_pct = min(100.0, (latest_icu / total_icu) * 100.0)

    # Anomaly detection
    anomalies = risk_engine.detect_outbreak_anomalies(cases)
    has_recent_anomaly = any(anomalies[-3:])

    # Early Warning Index
    ewi = risk_engine.compute_early_warning_index(
        current_rt=current_rt,
        weekly_case_growth_pct=growth_pct,
        test_positivity_rate=latest_tpr,
        icu_occupancy_pct=icu_pct,
        anomaly_flag=has_recent_anomaly
    )

    # Hospital exhaustion calculation
    new_daily_icu = int(cases[-1] * path_meta["hosp_rate"] * path_meta["icu_rate"])
    exhaustion = risk_engine.calculate_hospital_exhaustion(
        total_icu_beds=total_icu,
        current_occupied_icu=latest_icu,
        daily_new_icu_admissions=new_daily_icu
    )

    # Regional comparison snapshot for all regions
    all_regions_summary = []
    for r_name, r_info in REGIONS.items():
        r_df = df[(df["region"] == r_name) & (df["pathogen"] == pathogen)].sort_values("date").reset_index(drop=True)
        if r_df.empty:
            r_df = generate_epidemiological_timeseries(region=r_name, pathogen=pathogen)
        r_cases = r_df["daily_cases"].tolist()
        r_rt = risk_engine.estimate_rt(r_cases)[-1]
        r_growth = ((sum(r_cases[-7:]) - sum(r_cases[-14:-7])) / max(sum(r_cases[-14:-7]), 1)) * 100.0
        r_tpr = float(r_df["tpr_pct"].iloc[-1])
        r_icu_pct = min(100.0, (int(r_df["icu_active"].iloc[-1]) / r_info["icu_beds"]) * 100.0)
        r_ewi = risk_engine.compute_early_warning_index(r_rt, r_growth, r_tpr, r_icu_pct)
        all_regions_summary.append({
            "region": r_name,
            "lat": r_info["lat"],
            "lng": r_info["lng"],
            "rt": r_rt,
            "ewi_score": r_ewi["score"],
            "tier": r_ewi["tier"],
            "color": r_ewi["color"],
            "latest_cases": int(r_cases[-1]),
            "icu_pct": round(r_icu_pct, 1)
        })

    return jsonify({
        "region": region,
        "pathogen": pathogen,
        "metadata": {
            "population": reg_meta["population"],
            "total_icu_beds": total_icu,
            "total_acute_beds": reg_meta["acute_beds"],
            "transmission_type": path_meta["transmission_type"]
        },
        "kpis": {
            "latest_daily_cases": int(cases[-1]),
            "seven_day_avg_cases": int(round(recent_7_sum / 7.0)),
            "weekly_growth_pct": round(growth_pct, 1),
            "current_rt": current_rt,
            "latest_tpr_pct": latest_tpr,
            "active_hospitalized": int(sub_df["hospitalized_active"].iloc[-1]),
            "active_icu": latest_icu,
            "icu_occupancy_pct": round(icu_pct, 1),
            "vaccination_pct": float(sub_df["vaccination_coverage_pct"].iloc[-1]),
            "cumulative_deaths": int(sub_df["cumulative_deaths"].iloc[-1]),
            "ewi": ewi,
            "exhaustion": exhaustion,
            "recent_anomaly_detected": has_recent_anomaly
        },
        "timeseries": {
            "dates": dates[-60:],
            "daily_cases": cases[-60:],
            "rt": rt_series[-60:],
            "tpr": sub_df["tpr_pct"].tolist()[-60:],
            "hospitalized": sub_df["hospitalized_active"].tolist()[-60:],
            "icu": sub_df["icu_active"].tolist()[-60:],
            "mobility": sub_df["mobility_index"].tolist()[-60:],
            "anomalies": anomalies[-60:]
        },
        "all_regions_summary": all_regions_summary
    })

@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json() or {}
    region = data.get("region", "India")
    pathogen = data.get("pathogen", "COVID-19 (SARS-CoV-2)")
    horizon = int(data.get("horizon", 14))

    df = get_or_create_benchmark_dataset()
    sub_df = df[(df["region"] == region) & (df["pathogen"] == pathogen)].sort_values("date").reset_index(drop=True)
    if sub_df.empty:
        sub_df = generate_epidemiological_timeseries(region=region, pathogen=pathogen)

    model = get_forecaster()
    forecast_results = model.forecast(sub_df, horizon_days=horizon)

    # Estimate hospital requirements for projected cases
    path_meta = PATHOGENS.get(pathogen, PATHOGENS["COVID-19 (SARS-CoV-2)"])
    proj_hospitalized = [int(round(c * path_meta["hosp_rate"] * 3.5)) for c in forecast_results["predictions"]]
    proj_icu = [int(round(h * path_meta["icu_rate"])) for h in proj_hospitalized]

    return jsonify({
        "dates": forecast_results["dates"],
        "predicted_cases": forecast_results["predictions"],
        "lower_95_ci": forecast_results["lower_95_ci"],
        "upper_95_ci": forecast_results["upper_95_ci"],
        "projected_hospitalized": proj_hospitalized,
        "projected_icu": proj_icu,
        "horizon_days": horizon
    })

@app.route("/api/simulate", methods=["POST"])
def api_simulate():
    data = request.get_json() or {}
    
    population = int(data.get("population", 10_000_000))
    initial_infected = int(data.get("initial_infected", 150))
    r0 = float(data.get("r0", 2.5))
    days = int(data.get("days", 120))
    distancing = float(data.get("distancing", 0)) / 100.0
    masks = float(data.get("masks", 0)) / 100.0
    lockdown = float(data.get("lockdown", 0)) / 100.0
    vax_rate = float(data.get("vax_rate", 0.1)) / 100.0
    intervention_start = int(data.get("intervention_start", 25))

    # Baseline unmitigated run
    sim_baseline = SEIRModel(population=population, initial_infected=initial_infected)
    res_baseline = sim_baseline.simulate(
        days=days,
        r0=r0,
        distancing_reduction=0.0,
        mask_mandate_reduction=0.0,
        lockdown_reduction=0.0,
        vaccination_rate_daily=0.0002
    )

    # Intervened run
    sim_intervened = SEIRModel(population=population, initial_infected=initial_infected)
    res_intervened = sim_intervened.simulate(
        days=days,
        r0=r0,
        distancing_reduction=distancing,
        mask_mandate_reduction=masks,
        lockdown_reduction=lockdown,
        vaccination_rate_daily=vax_rate,
        intervention_start_day=intervention_start
    )

    deaths_averted = max(0, res_baseline["total_deaths"] - res_intervened["total_deaths"])
    peak_reduction_pct = max(0.0, ((res_baseline["peak_infected"] - res_intervened["peak_infected"]) / res_baseline["peak_infected"]) * 100.0)

    return jsonify({
        "baseline": res_baseline,
        "intervened": res_intervened,
        "impact": {
            "deaths_averted": deaths_averted,
            "peak_reduction_pct": round(peak_reduction_pct, 1),
            "peak_day_shift": res_intervened["peak_day"] - res_baseline["peak_day"],
            "final_rt": res_intervened["rt"][-1]
        }
    })

@app.route("/api/briefing")
def api_briefing():
    region = request.args.get("region", "India")
    pathogen = request.args.get("pathogen", "COVID-19 (SARS-CoV-2)")
    
    df = get_or_create_benchmark_dataset()
    sub_df = df[(df["region"] == region) & (df["pathogen"] == pathogen)].sort_values("date").reset_index(drop=True)
    if sub_df.empty:
        sub_df = generate_epidemiological_timeseries(region=region, pathogen=pathogen)

    reg_meta = REGIONS.get(region, REGIONS["India"])
    path_meta = PATHOGENS.get(pathogen, PATHOGENS["COVID-19 (SARS-CoV-2)"])

    cases = sub_df["daily_cases"].tolist()
    rt_series = risk_engine.estimate_rt(cases)
    current_rt = rt_series[-1]

    recent_7_sum = sum(cases[-7:])
    prev_7_sum = sum(cases[-14:-7]) if len(cases) >= 14 else recent_7_sum
    growth_pct = ((recent_7_sum - prev_7_sum) / max(prev_7_sum, 1)) * 100.0

    tpr = float(sub_df["tpr_pct"].iloc[-1])
    icu_active = int(sub_df["icu_active"].iloc[-1])
    icu_pct = min(100.0, (icu_active / reg_meta["icu_beds"]) * 100.0)

    ewi = risk_engine.compute_early_warning_index(current_rt, growth_pct, tpr, icu_pct)
    exhaustion = risk_engine.calculate_hospital_exhaustion(
        total_icu_beds=reg_meta["icu_beds"],
        current_occupied_icu=icu_active,
        daily_new_icu_admissions=int(cases[-1] * path_meta["hosp_rate"] * path_meta["icu_rate"])
    )

    # Generate structured epidemiological intelligence memo
    briefing = {
        "title": f"EPIDEMIOLOGICAL SURVEILLANCE & EARLY WARNING BRIEFING: {region.upper()}",
        "timestamp": pd.Timestamp.now().strftime("%B %d, %Y - %H:%M UTC"),
        "threat_level": ewi["tier"],
        "threat_color": ewi["color"],
        "ewi_score": ewi["score"],
        "summary": (
            f"Surveillance data indicates an active {ewi['tier']} risk posture for {pathogen} in {region}. "
            f"The effective reproduction number is estimated at Rt = {current_rt:.2f}, indicating "
            f"{'expanding transmission (Rt > 1.0)' if current_rt > 1.0 else 'receding transmission (Rt < 1.0)'}. "
            f"Seven-day case velocity registered at {growth_pct:+.1f}% with a diagnostic test positivity rate (TPR) of {tpr:.1f}%."
        ),
        "hospital_status": (
            f"Regional ICU occupancy is at {icu_pct:.1f}% ({icu_active:,} / {reg_meta['icu_beds']:,} beds). "
            f"Capacity stress status: {exhaustion['status']}. "
            f"Estimated days to ICU saturation: {exhaustion['days_to_exhaustion']}."
        ),
        "tactical_recommendations": [
            ewi["recommended_action"],
            f"Targeted genomic sequencing and contact tracing across top metropolitan clusters.",
            f"Maintain testing throughput above {int(cases[-1] * 15):,} tests/day to keep TPR below 5.0%.",
            f"Health resource prepositioning: reserve supplemental oxygen and mobilize surge nursing personnel."
        ]
    }
    return jsonify(briefing)

if __name__ == "__main__":
    print("Initializing PandemicWatch AI Command Center...")
    # Pre-train / verify model availability
    get_forecaster()
    print("Server running on http://127.0.0.1:5000")
    app.run(host="0.0.0.0", port=5000, debug=True)
