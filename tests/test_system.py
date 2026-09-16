"""
Unit and Integration Tests for PandemicWatch AI
Covers:
- SEIR compartmental ODE integration and intervention dynamics
- Epidemic Risk Engine (Rt estimation, Early Warning Index, ICU stress)
- Machine Learning Forecaster inference and confidence bounds
- Flask REST API endpoints
"""

import os
import sys
import unittest
import json
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from models.seir_model import SEIRModel
from models.risk_engine import EpidemicRiskEngine
from models.ml_forecaster import EpidemicForecaster
from data.generator import generate_epidemiological_timeseries
from app import app

class TestSEIRModel(unittest.TestCase):
    def setUp(self):
        self.model = SEIRModel(population=1_000_000, initial_infected=50, initial_exposed=100)

    def test_simulation_shape_and_conservation(self):
        res = self.model.simulate(days=60, r0=2.5)
        self.assertEqual(len(res["days"]), 61)
        self.assertEqual(len(res["infected"]), 61)
        
        # Test population conservation approximately holds (S + E + I + R + D ~= N)
        total_pop_final = (
            res["susceptible"][-1] +
            res["exposed"][-1] +
            res["infected"][-1] +
            res["recovered"][-1] +
            res["deceased"][-1]
        )
        self.assertAlmostEqual(total_pop_final, 1_000_000, delta=50)

    def test_interventions_flatten_curve(self):
        baseline = self.model.simulate(days=90, r0=2.8, distancing_reduction=0.0)
        intervened = self.model.simulate(
            days=90,
            r0=2.8,
            distancing_reduction=0.4,
            mask_mandate_reduction=0.3,
            intervention_start_day=15
        )
        self.assertLess(intervened["peak_infected"], baseline["peak_infected"])
        self.assertLessEqual(intervened["total_deaths"], baseline["total_deaths"])


class TestRiskEngine(unittest.TestCase):
    def setUp(self):
        self.engine = EpidemicRiskEngine(serial_interval_days=5.0)

    def test_rt_estimation(self):
        # Exponentially growing case counts
        growing_cases = [100 * (1.1 ** i) for i in range(30)]
        rt_series = self.engine.estimate_rt(growing_cases)
        self.assertEqual(len(rt_series), 30)
        self.assertGreater(rt_series[-1], 1.2)

        # Declining case counts
        declining_cases = [1000 * (0.9 ** i) for i in range(30)]
        rt_declining = self.engine.estimate_rt(declining_cases)
        self.assertLess(rt_declining[-1], 1.0)

    def test_early_warning_index_bounds_and_tiers(self):
        low_risk = self.engine.compute_early_warning_index(
            current_rt=0.75,
            weekly_case_growth_pct=-15.0,
            test_positivity_rate=2.1,
            icu_occupancy_pct=28.0
        )
        self.assertIn(low_risk["tier"], ["LOW", "MODERATE"])
        self.assertTrue(0.0 <= low_risk["score"] <= 100.0)

        high_risk = self.engine.compute_early_warning_index(
            current_rt=2.1,
            weekly_case_growth_pct=85.0,
            test_positivity_rate=14.5,
            icu_occupancy_pct=92.0
        )
        self.assertIn(high_risk["tier"], ["HIGH", "CRITICAL"])
        self.assertGreater(high_risk["score"], 60.0)

    def test_hospital_exhaustion_calculation(self):
        # Deficit surge: 120 new admissions vs 95 daily discharges with 50 available beds
        res = self.engine.calculate_hospital_exhaustion(
            total_icu_beds=1000,
            current_occupied_icu=950,
            daily_new_icu_admissions=120,
            mean_los_days=10.0
        )
        self.assertIn(res["status"], ["CRITICAL_SURGE", "WARNING_SURGE"])
        self.assertIsInstance(res["days_to_exhaustion"], (int, float))
        self.assertLessEqual(res["days_to_exhaustion"], 5)


class TestMLForecaster(unittest.TestCase):
    def test_inference_and_confidence_bounds(self):
        model_path = os.path.join(BASE_DIR, "models", "xgboost_forecaster.joblib")
        self.assertTrue(os.path.exists(model_path), "Trained model joblib file must exist.")
        
        forecaster = EpidemicForecaster.load(model_path)
        sample_df = generate_epidemiological_timeseries(region="India", pathogen="COVID-19 (SARS-CoV-2)", num_days=90)
        
        horizon = 14
        pred = forecaster.forecast(sample_df, horizon_days=horizon)
        
        self.assertEqual(len(pred["predictions"]), horizon)
        self.assertEqual(len(pred["lower_95_ci"]), horizon)
        self.assertEqual(len(pred["upper_95_ci"]), horizon)
        
        for i in range(horizon):
            self.assertGreaterEqual(pred["predictions"][i], 0)
            self.assertGreaterEqual(pred["upper_95_ci"][i], pred["lower_95_ci"][i])


class TestFlaskAPI(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_index_page(self):
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)
        self.assertIn(b"PandemicWatch", res.data)

    def test_api_overview(self):
        res = self.client.get("/api/overview?region=India&pathogen=COVID-19 (SARS-CoV-2)")
        self.assertEqual(res.status_code, 200)
        data = json.loads(res.data)
        self.assertIn("kpis", data)
        self.assertIn("timeseries", data)
        self.assertIn("all_regions_summary", data)
        self.assertGreater(len(data["timeseries"]["dates"]), 0)

    def test_api_predict(self):
        payload = {"region": "India", "pathogen": "COVID-19 (SARS-CoV-2)", "horizon": 7}
        res = self.client.post("/api/predict", data=json.dumps(payload), content_type="application/json")
        self.assertEqual(res.status_code, 200)
        data = json.loads(res.data)
        self.assertEqual(len(data["predicted_cases"]), 7)
        self.assertEqual(len(data["projected_icu"]), 7)

    def test_api_simulate(self):
        payload = {
            "population": 5_000_000,
            "r0": 2.6,
            "distancing": 30,
            "masks": 40,
            "lockdown": 0,
            "vax_rate": 0.2,
            "days": 60
        }
        res = self.client.post("/api/simulate", data=json.dumps(payload), content_type="application/json")
        self.assertEqual(res.status_code, 200)
        data = json.loads(res.data)
        self.assertIn("impact", data)
        self.assertIn("baseline", data)
        self.assertIn("intervened", data)

    def test_api_briefing(self):
        res = self.client.get("/api/briefing?region=India")
        self.assertEqual(res.status_code, 200)
        data = json.loads(res.data)
        self.assertIn("threat_level", data)
        self.assertIn("tactical_recommendations", data)

if __name__ == "__main__":
    unittest.main()
