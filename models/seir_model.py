"""
Epidemiological SEIR (Susceptible-Exposed-Infectious-Recovered) Model
with Public Health Interventions, Vaccination Dynamics, and Hospital Load Projections.
"""

import numpy as np
from scipy.integrate import odeint

class SEIRModel:
    """
    Extended SEIR Compartmental Model for infectious disease modeling.
    Includes:
    - Incubation period (Exposed compartment E)
    - Transmission modulation via non-pharmaceutical interventions (NPIs)
    - Vaccination dynamics (Susceptible -> Recovered transition)
    - Hospitalization and ICU burden estimates
    - Disease mortality (Deceased compartment D)
    """

    def __init__(self, population=10_000_000, initial_infected=100, initial_exposed=200):
        self.N = population
        self.I0 = initial_infected
        self.E0 = initial_exposed
        self.R0_init = 0
        self.D0 = 0
        self.S0 = self.N - self.I0 - self.E0

    @staticmethod
    def deriv(y, t, N, beta_func, sigma, gamma, mu, nu):
        """
        ODE system:
        dS/dt = - beta(t) * S * I / N - nu * S
        dE/dt =   beta(t) * S * I / N - sigma * E
        dI/dt =   sigma * E - gamma * I
        dRdt =   (1.0 - mu) * gamma * I + nu * S
        dDdt =   mu * gamma * I
        """
        S, E, I, R, D = y
        beta = beta_func(t)
        
        dSdt = - (beta * S * I) / N - nu * S
        dEdt =   (beta * S * I) / N - sigma * E
        dIdt =   sigma * E - gamma * I
        dRdt =   (1.0 - mu) * gamma * I + nu * S
        dDdt =   mu * gamma * I
        
        return [dSdt, dEdt, dIdt, dRdt, dDdt]

    def simulate(
        self,
        days=120,
        r0=2.5,
        incubation_days=5.2,
        infectious_days=7.0,
        fatality_rate=0.012,
        vaccination_rate_daily=0.001,
        intervention_start_day=20,
        distancing_reduction=0.0,
        mask_mandate_reduction=0.0,
        lockdown_reduction=0.0,
        hospitalization_rate=0.05,
        icu_rate_of_hospitalized=0.20
    ):
        """
        Run forward simulation using ODE integration.
        
        Parameters:
        - days: Simulation horizon (in days)
        - r0: Basic Reproduction Number (R0 = beta / gamma)
        - incubation_days: 1 / sigma
        - infectious_days: 1 / gamma
        - fatality_rate: mu (proportion of resolved cases that result in death)
        - vaccination_rate_daily: nu (daily fraction of susceptible population vaccinated)
        - intervention_start_day: day at which NPIs take effect
        - distancing_reduction: 0.0 to 1.0 (relative contact reduction)
        - mask_mandate_reduction: 0.0 to 1.0 (transmission reduction from masks)
        - lockdown_reduction: 0.0 to 1.0 (mobility drop reduction)
        - hospitalization_rate: proportion of active infected requiring acute bed care
        - icu_rate_of_hospitalized: proportion of hospitalized cases requiring ICU beds
        """
        gamma = 1.0 / infectious_days
        sigma = 1.0 / incubation_days
        beta_base = r0 * gamma

        # Cumulative reduction factor from interventions
        combined_reduction = min(
            0.92,
            1.0 - (1.0 - distancing_reduction * 0.5) * 
                  (1.0 - mask_mandate_reduction * 0.35) * 
                  (1.0 - lockdown_reduction * 0.75)
        )

        def beta_func(t):
            if t < intervention_start_day:
                return beta_base
            else:
                return beta_base * (1.0 - combined_reduction)

        t = np.linspace(0, days, days + 1)
        y0 = [self.S0, self.E0, self.I0, self.R0_init, self.D0]

        ret = odeint(
            self.deriv,
            y0,
            t,
            args=(self.N, beta_func, sigma, gamma, fatality_rate, vaccination_rate_daily)
        )
        S, E, I, R, D = ret.T

        # Derive daily new incident cases (sigma * E)
        daily_new_cases = np.maximum(0, sigma * E)
        
        # Derive healthcare utilization
        hospitalized = I * hospitalization_rate
        icu = hospitalized * icu_rate_of_hospitalized

        # Effective Rt over time: Rt = (beta(t)/gamma) * (S(t)/N)
        rt_series = [(beta_func(step) / gamma) * (S[i] / self.N) for i, step in enumerate(t)]

        return {
            "days": [int(x) for x in t],
            "susceptible": [int(round(x)) for x in S],
            "exposed": [int(round(x)) for x in E],
            "infected": [int(round(x)) for x in I],
            "recovered": [int(round(x)) for x in R],
            "deceased": [int(round(x)) for x in D],
            "daily_new_cases": [int(round(x)) for x in daily_new_cases],
            "hospitalized": [int(round(x)) for x in hospitalized],
            "icu": [int(round(x)) for x in icu],
            "rt": [float(round(x, 3)) for x in rt_series],
            "peak_infected": int(round(np.max(I))),
            "peak_day": int(t[np.argmax(I)]),
            "total_deaths": int(round(D[-1])),
            "attack_rate_percent": float(round(((self.N - S[-1]) / self.N) * 100, 2))
        }
