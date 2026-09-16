"""
Epidemiological Benchmark Dataset Generator & Curator
Provides multi-region, multi-pathogen surveillance data for model training,
benchmarking, and dashboard visualization.
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

REGIONS = {
    "India": {
        "population": 1_400_000_000,
        "icu_beds": 95_000,
        "acute_beds": 700_000,
        "baseline_mobility": 85,
        "lat": 20.5937,
        "lng": 78.9629,
        "subregions": ["Maharashtra", "Kerala", "Delhi", "Karnataka", "Tamil Nadu"]
    },
    "United States": {
        "population": 331_000_000,
        "icu_beds": 85_000,
        "acute_beds": 520_000,
        "baseline_mobility": 90,
        "lat": 37.0902,
        "lng": -95.7129,
        "subregions": ["California", "New York", "Texas", "Florida", "Illinois"]
    },
    "United Kingdom": {
        "population": 67_000_000,
        "icu_beds": 5_500,
        "acute_beds": 130_000,
        "baseline_mobility": 88,
        "lat": 55.3781,
        "lng": -3.4360,
        "subregions": ["London", "South East", "North West", "West Midlands", "Scotland"]
    },
    "Brazil": {
        "population": 214_000_000,
        "icu_beds": 42_000,
        "acute_beds": 310_000,
        "baseline_mobility": 80,
        "lat": -14.2350,
        "lng": -51.9253,
        "subregions": ["São Paulo", "Rio de Janeiro", "Minas Gerais", "Bahia", "Paraná"]
    }
}

PATHOGENS = {
    "COVID-19 (SARS-CoV-2)": {
        "r0": 2.8,
        "incubation_days": 4.5,
        "infectious_days": 6.0,
        "hosp_rate": 0.045,
        "icu_rate": 0.22,
        "fatality_rate": 0.012,
        "transmission_type": "Respiratory droplet & airborne"
    },
    "Dengue Virus (DEN-1-4)": {
        "r0": 2.1,
        "incubation_days": 5.5,
        "infectious_days": 5.0,
        "hosp_rate": 0.065,
        "icu_rate": 0.15,
        "fatality_rate": 0.008,
        "transmission_type": "Vector-borne (Aedes mosquito)"
    },
    "Novel Influenza (H5N1 Flu)": {
        "r0": 1.9,
        "incubation_days": 2.5,
        "infectious_days": 4.5,
        "hosp_rate": 0.075,
        "icu_rate": 0.28,
        "fatality_rate": 0.025,
        "transmission_type": "Respiratory droplet & contact"
    }
}

def generate_epidemiological_timeseries(
    region="India",
    pathogen="COVID-19 (SARS-CoV-2)",
    num_days=180,
    seed=42
):
    """
    Generates realistic, epidemiological time-series surveillance data
    with wave peaks, testing variability, non-pharmaceutical interventions,
    weather seasonality, and hospital admissions.
    """
    np.random.seed(seed)
    reg_meta = REGIONS.get(region, REGIONS["India"])
    path_meta = PATHOGENS.get(pathogen, PATHOGENS["COVID-19 (SARS-CoV-2)"])
    
    start_date = datetime(2025, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(num_days)]

    # Outbreak wave dynamics: combination of baseline, surge peak 1, and surge peak 2
    t = np.linspace(0, num_days, num_days)
    
    # Wave 1 peak around day 65, Wave 2 peak around day 135
    peak1 = 35000 * np.exp(-((t - 65) ** 2) / (2 * 18 ** 2))
    peak2 = 55000 * np.exp(-((t - 135) ** 2) / (2 * 14 ** 2))
    base_cases = 1200 + 400 * np.sin(2 * np.pi * t / 60)
    
    # Poisson-like observational noise and day-of-week reporting lag (e.g. lower weekend counts)
    day_of_week_factor = np.array([0.78 if d.weekday() == 6 else 0.88 if d.weekday() == 5 else 1.05 for d in dates])
    noise = np.random.normal(1.0, 0.06, num_days)

    scale_factor = (reg_meta["population"] / 1_400_000_000) ** 0.5
    raw_cases = (base_cases + peak1 + peak2) * scale_factor * day_of_week_factor * noise
    daily_cases = np.maximum(50, raw_cases).astype(int)

    # Testing capacity and test positivity rate
    base_tests = int(daily_cases.max() * 18)
    daily_tests = (base_tests + daily_cases * 6 + np.random.normal(0, 1000, num_days)).astype(int)
    daily_tests = np.maximum(daily_cases + 500, daily_tests)
    tpr = np.round((daily_cases / daily_tests) * 100.0, 2)

    # Hospitalization and ICU burden (lagged by ~5 days and ~8 days from case onset)
    hosp_lag = 5
    icu_lag = 8
    hosp_active = np.zeros(num_days)
    icu_active = np.zeros(num_days)

    for i in range(num_days):
        recent_cases_hosp = daily_cases[max(0, i-14):max(0, i-hosp_lag+1)]
        recent_cases_icu = daily_cases[max(0, i-20):max(0, i-icu_lag+1)]
        hosp_active[i] = np.sum(recent_cases_hosp) * path_meta["hosp_rate"] * 0.45
        icu_active[i] = np.sum(recent_cases_icu) * path_meta["hosp_rate"] * path_meta["icu_rate"] * 0.55

    hosp_active = hosp_active.astype(int)
    icu_active = icu_active.astype(int)

    # Deaths (lagged by 12-16 days)
    daily_deaths = np.zeros(num_days, dtype=int)
    for i in range(num_days):
        if i >= 14:
            mort = daily_cases[i-14] * path_meta["fatality_rate"] * np.random.normal(1.0, 0.08)
            daily_deaths[i] = max(0, int(round(mort)))

    # Interventions & Mobility
    # Mobility drops when cases surge (voluntary distancing or government restrictions)
    case_pressure = daily_cases / daily_cases.max()
    mobility = np.clip(reg_meta["baseline_mobility"] - (case_pressure * 38) + np.random.normal(0, 2, num_days), 35, 100)
    stringency = np.clip(15 + (case_pressure * 75) + np.random.normal(0, 3, num_days), 0, 100)

    # Vaccination coverage (gradual S-curve)
    vax_coverage = 45.0 + 35.0 / (1.0 + np.exp(-(t - 80) / 25))
    vax_coverage = np.round(np.clip(vax_coverage, 0.0, 95.0), 1)

    # Environmental indicators
    temp = np.round(24.0 + 8.0 * np.sin(2 * np.pi * t / 365) + np.random.normal(0, 1.5, num_days), 1)
    humidity = np.round(65.0 + 15.0 * np.cos(2 * np.pi * t / 365) + np.random.normal(0, 3.0, num_days), 1)

    df = pd.DataFrame({
        "date": [d.strftime("%Y-%m-%d") for d in dates],
        "region": region,
        "pathogen": pathogen,
        "daily_cases": daily_cases,
        "cumulative_cases": np.cumsum(daily_cases),
        "daily_deaths": daily_deaths,
        "cumulative_deaths": np.cumsum(daily_deaths),
        "daily_tests": daily_tests,
        "tpr_pct": tpr,
        "hospitalized_active": hosp_active,
        "icu_active": icu_active,
        "mobility_index": np.round(mobility, 1),
        "stringency_index": np.round(stringency, 1),
        "vaccination_coverage_pct": vax_coverage,
        "temperature_c": temp,
        "humidity_pct": humidity
    })

    return df

def get_or_create_benchmark_dataset():
    """Returns multi-region combined dataset."""
    os.makedirs(DATA_DIR, exist_ok=True)
    csv_path = os.path.join(DATA_DIR, "epidemic_surveillance_benchmark.csv")
    
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)

    dfs = []
    for region in REGIONS.keys():
        for pathogen in PATHOGENS.keys():
            df = generate_epidemiological_timeseries(region=region, pathogen=pathogen)
            dfs.append(df)

    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df.to_csv(csv_path, index=False)
    print(f"Generated and saved benchmark dataset: {csv_path} ({len(combined_df)} records)")
    return combined_df

if __name__ == "__main__":
    df = get_or_create_benchmark_dataset()
    print("Dataset preview:")
    print(df.head())
