"""
generate_dataset.py
--------------------
Generates a physics-informed synthetic dataset for PV module degradation.

Degradation is modeled as a combination of three well-established
reliability physics mechanisms, plus realistic measurement noise:

1. Arrhenius kinetics (thermal acceleration of chemical degradation,
   e.g. encapsulant discoloration / EVA browning):
        k_T = A * exp(-Ea / (k_B * T))

2. Coffin-Manson fatigue (thermo-mechanical fatigue of solder joints
   from repeated thermal cycling):
        N_f = C * (delta_T) ^ (-n)

3. Peck's humidity model (moisture-driven corrosion / delamination):
        k_RH = B * (RH) ^ m * exp(-Ea_h / (k_B * T))

The three mechanisms are combined into a single annual degradation
rate (%/year), then integrated over module age with stochastic noise
to produce a realistic tabular dataset suitable for supervised
regression (predicting cumulative power degradation) and a derived
binary "fault flag" target for classification-style evaluation.
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)

BOLTZMANN_EV = 8.617333262e-5  # eV/K

N_SAMPLES = 2000


def arrhenius_rate(temp_k, ea_ev, a_pref):
    return a_pref * np.exp(-ea_ev / (BOLTZMANN_EV * temp_k))


def coffin_manson_cycles(delta_t, c_const=1.0, n_exp=2.2):
    delta_t = np.clip(delta_t, 1, None)
    return c_const * delta_t ** (-n_exp)


def peck_rate(rh_pct, temp_k, ea_ev, b_pref, m_exp=2.7):
    rh_frac = np.clip(rh_pct, 1, 100) / 100.0
    return b_pref * (rh_frac ** m_exp) * np.exp(-ea_ev / (BOLTZMANN_EV * temp_k))


def generate():
    module_age_years = RNG.uniform(0.5, 25, N_SAMPLES)
    ambient_temp_c = RNG.normal(28, 8, N_SAMPLES).clip(-5, 50)
    module_temp_c = ambient_temp_c + RNG.normal(22, 6, N_SAMPLES).clip(5, 45)  # NOCT-style uplift
    relative_humidity = RNG.uniform(15, 95, N_SAMPLES)
    daily_temp_swing = RNG.normal(18, 7, N_SAMPLES).clip(2, 45)  # thermal cycling amplitude
    irradiance_kwh_m2_day = RNG.normal(5.2, 1.1, N_SAMPLES).clip(2, 8)
    soiling_index = RNG.uniform(0, 1, N_SAMPLES)  # 0 = clean, 1 = heavily soiled
    module_temp_k = module_temp_c + 273.15

    # --- physics-based degradation contributions ---
    thermal_chem_rate = arrhenius_rate(module_temp_k, ea_ev=0.7, a_pref=3.2e6)
    fatigue_rate = 1.0 / coffin_manson_cycles(daily_temp_swing, c_const=2.0e4, n_exp=2.2)
    humidity_rate = peck_rate(relative_humidity, module_temp_k, ea_ev=0.4, b_pref=1.1e4)

    # normalize each mechanism to a %/year scale, then weight by known
    # literature-informed contribution shares for c-Si PERC modules
    thermal_chem_pct_yr = (thermal_chem_rate / thermal_chem_rate.mean()) * 0.28
    fatigue_pct_yr = (fatigue_rate / fatigue_rate.mean()) * 0.14
    humidity_pct_yr = (humidity_rate / humidity_rate.mean()) * 0.10

    soiling_penalty_pct_yr = soiling_index * 0.15
    irradiance_stress = (irradiance_kwh_m2_day / irradiance_kwh_m2_day.mean()) * 0.05

    annual_degradation_rate = (
        thermal_chem_pct_yr + fatigue_pct_yr + humidity_pct_yr
        + soiling_penalty_pct_yr + irradiance_stress
    )
    annual_degradation_rate += RNG.normal(0, 0.11, N_SAMPLES)  # sensor/measurement noise
    annual_degradation_rate = np.clip(annual_degradation_rate, 0.05, 3.0)

    # cumulative power loss compounds non-linearly with age (slight acceleration)
    cumulative_degradation_pct = (
        annual_degradation_rate * module_age_years
        + 0.01 * module_age_years ** 1.6
        + RNG.normal(0, 1.65, N_SAMPLES)
    )
    cumulative_degradation_pct = np.clip(cumulative_degradation_pct, 0, 45)

    current_power_ratio = np.clip(1 - cumulative_degradation_pct / 100, 0.4, 1.0)

    # Fault flag: IEC 61215 style threshold heuristic -> module considered
    # "at fault / underperforming" if degradation exceeds warranty curve
    # (linear 0.7%/year, ~80% at year 25) by a margin
    warranty_limit_pct = 2.5 + 0.7 * module_age_years
    fault_flag = (cumulative_degradation_pct > warranty_limit_pct).astype(int)

    df = pd.DataFrame({
        "module_age_years": module_age_years.round(2),
        "ambient_temp_c": ambient_temp_c.round(2),
        "module_temp_c": module_temp_c.round(2),
        "relative_humidity_pct": relative_humidity.round(2),
        "daily_temp_swing_c": daily_temp_swing.round(2),
        "irradiance_kwh_m2_day": irradiance_kwh_m2_day.round(2),
        "soiling_index": soiling_index.round(3),
        "annual_degradation_rate_pct": annual_degradation_rate.round(4),
        "cumulative_degradation_pct": cumulative_degradation_pct.round(3),
        "current_power_ratio": current_power_ratio.round(4),
        "fault_flag": fault_flag,
    })
    return df


if __name__ == "__main__":
    df = generate()
    df.to_csv("/home/claude/pv-degradation-predictor/data/pv_degradation_dataset.csv", index=False)
    print(df.head())
    print("\nShape:", df.shape)
    print("\nFault rate:", df.fault_flag.mean().round(3))
