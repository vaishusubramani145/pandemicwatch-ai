# 🦠 PandemicWatch AI
### Final Internship Machine Learning Project in Public Health & Epidemiology
**Author:** S. Vaishnavi  
**Domain:** Artificial Intelligence & Machine Learning in Healthcare / Computational Epidemiology  
**Tech Stack:** Python 3.13, XGBoost, Scikit-Learn, SciPy ODE, Pandas, NumPy, Flask, Chart.js, Leaflet.js  

---

## 📌 Executive Summary

**PandemicWatch AI** is an end-to-end intelligent epidemiological surveillance, outbreak forecasting, and public health early warning system. Designed as a capstone internship project, it bridges classical compartmental epidemiology with modern machine learning:

1. **Epidemiological Modeling (SEIR ODE Engine):** Simulates susceptible, exposed, infectious, and recovered population dynamics with time-varying non-pharmaceutical interventions (masking, distancing, circuit-breaker lockdowns) and vaccination rollouts.
2. **Machine Learning Time-Series Forecaster (XGBoost):** Recursive multi-horizon case projections (7, 14, 30 days) with 95% confidence bounds, trained on multi-regional surveillance features.
3. **Dynamic Epidemic Metrics ($R_t$ & EWI):** Real-time effective reproduction number calculation and a composite Early Warning Index (0–100) scoring regional alert tiers.
4. **Healthcare Resource Stress Engine:** Projects acute and ICU bed capacity saturation timelines based on length-of-stay distributions.
5. **Automated Situation Reports (SitRep):** AI-synthesized public health briefings with actionable tactical recommendations for health authorities.

---

## 🔬 Mathematical & Epidemiological Foundations

### 1. SEIR Compartmental System with Interventions
The disease transmission dynamics are governed by the following system of non-linear ordinary differential equations (ODEs):

$$\frac{dS}{dt} = -\beta(t) \frac{S \cdot I}{N} - \nu \cdot S$$

$$\frac{dE}{dt} = \beta(t) \frac{S \cdot I}{N} - \sigma \cdot E$$

$$\frac{dI}{dt} = \sigma \cdot E - \gamma \cdot I$$

$$\frac{dR}{dt} = (1 - \mu) \cdot \gamma \cdot I + \nu \cdot S$$

$$\frac{dD}{dt} = \mu \cdot \gamma \cdot I$$

Where:
- $N$: Total population size
- $\sigma = \frac{1}{\text{incubation period}}$ (rate of progression from exposed to infectious)
- $\gamma = \frac{1}{\text{infectious duration}}$ (rate of recovery)
- $\mu$: Infection fatality rate (IFR)
- $\nu$: Daily fraction of population receiving effective immunization
- $\beta(t)$: Time-varying transmission parameter modulated by policy interventions:

$$\beta(t) = \beta_0 \cdot \left[1 - \min\left(0.92, 1 - (1 - 0.5 \cdot d)(1 - 0.35 \cdot m)(1 - 0.75 \cdot \ell)\right)\right]$$

*(where $d$ = distancing adherence, $m$ = mask adherence, and $\ell$ = lockdown stringency)*

---

### 2. Effective Reproduction Number ($R_t$)
Real-time transmission velocity is estimated using the intrinsic growth rate $r$ and generation interval $T_c$:

$$r(t) = \frac{1}{w} \ln\left(\frac{\overline{C}_t}{\overline{C}_{t-w}}\right)$$

$$R_t = \exp(r(t) \cdot T_c)$$

- **$R_t > 1.0$**: Epidemic is undergoing exponential propagation.
- **$R_t < 1.0$**: Epidemic transmission is contracting and under control.

---

### 3. Composite Early Warning Index (EWI)
The EWI synthesizes four independent surveillance signals into a normalized 0–100 threat score:

$$\text{EWI} = 0.30 \cdot S_{R_t} + 0.25 \cdot S_{\Delta \text{cases}} + 0.20 \cdot S_{\text{TPR}} + 0.20 \cdot S_{\text{ICU}} + 0.05 \cdot S_{\text{anomaly}}$$

| Threat Tier | EWI Range | Color | Operational Action |
|---|---|---|---|
| **LOW** | 0.0 – 29.9 | 🟢 Green | Routine surveillance and standard sentinel testing |
| **MODERATE** | 30.0 – 54.9 | 🟡 Amber | Enhanced contact tracing, localized cluster isolation |
| **HIGH** | 55.0 – 74.9 | 🟠 Orange | Surge bed activation, indoor masking advisories |
| **CRITICAL** | 75.0 – 100.0 | 🔴 Red | Emergency surge mobilization, targeted gathering caps |

---

## 🤖 Machine Learning Forecaster

### Feature Engineering
- **Autoregressive Lags:** $t-1, t-2, t-3, t-7, t-14$ daily case counts
- **Rolling Statistics:** 7-day and 14-day rolling means and standard deviations
- **Growth Velocity Ratios:** $\frac{\text{MA}_7 + 1}{\text{MA}_{14} + 1}$
- **Surveillance Signals:** Diagnostic Test Positivity Rate (TPR) and 7-day lag
- **Exogenous Variables:** Community mobility index, Oxford stringency index, vaccination coverage %, ambient temperature, and humidity
- **Seasonality:** Day-of-week cyclical encoding and weekend indicator

### Model Evaluation Benchmarks

| Model Architecture | Validation $R^2$ | RMSE | MAE | MAPE (%) |
|---|---|---|---|---|
| **XGBoost Regressor (Production)** | **0.9979** | **354.51** | **263.57** | **3.19%** |
| Random Forest Regressor (Benchmark) | 0.9903 | 772.11 | 506.17 | 4.34% |

---

## 📂 Project Structure

```
pandemicwatch-ai/
├── app.py                      # Flask Application & REST API
├── train.py                    # Model Training, Evaluation & Serialization
├── models/
│   ├── seir_model.py           # SEIR ODE Solver & Policy Intervention Simulator
│   ├── ml_forecaster.py        # XGBoost & Random Forest Forecaster with 95% CI
│   ├── risk_engine.py          # Rt Estimator, EWI, Hospital Stress & Anomaly Guard
│   ├── xgboost_forecaster.joblib # Serialized Production Model
│   └── random_forest_forecaster.joblib # Serialized Benchmark Model
├── data/
│   ├── generator.py            # Multi-region, multi-pathogen surveillance data generator
│   └── epidemic_surveillance_benchmark.csv # Benchmark dataset
├── templates/
│   └── index.html              # Command Center Dashboard Template
├── static/
│   ├── css/style.css           # High-tech clinical UI styling
│   └── js/dashboard.js         # Chart.js & Leaflet interactive frontend controller
└── tests/
    └── test_system.py          # Comprehensive test suite (11 unit & integration tests)
```

---

## 🚀 Installation & Running

### Prerequisites
- Python 3.10+
- Installed libraries: `flask`, `scikit-learn`, `xgboost`, `pandas`, `numpy`, `scipy`, `matplotlib`

### 1. Run Unit & Integration Tests
Verify all 11 test assertions (differential equations conservation, risk scoring, ML inference, and REST endpoints):
```bash
python -m unittest tests/test_system.py
```

### 2. Train / Benchmark ML Models
```bash
python train.py
```

### 3. Launch the Web Application
```bash
python app.py
```
Open your browser and navigate to:
```
http://127.0.0.1:5000
```

---

## 📡 REST API Reference

- `GET /api/overview?region=<name>&pathogen=<name>`: Returns 60-day historical surveillance metrics, active KPIs, $R_t$, EWI score, and cross-regional risk matrix.
- `POST /api/predict`: Generates XGBoost recursive case projections with 95% confidence bands and projected ICU beds.
- `POST /api/simulate`: Executes SEIR differential equation integration given user-tuned intervention parameters.
- `GET /api/briefing?region=<name>&pathogen=<name>`: Returns a structured Epidemiological Situation Briefing (SitRep) for health authorities.
