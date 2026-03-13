# NHANES 2013-2014 Circadian Sleep Analysis

Actigraphy-based physical modeling of sleep-wake dynamics: Markov transition probability (Night P₀₁), Kramers potential well, and multivariate logistic regression with BMI/PHQ-9.

## Data (NHANES 2013-2014)

Place in project root:
- `DEMO_H.xpt`, `SLQ_H.xpt` — Required
- `PAXHR_H.xpt` — ~145MB, download from [NHANES](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2013/DataFiles/PAXHR_H.xpt)
- `BMX_H.xpt`, `DPQ_H.xpt` — For BMI/PHQ-9 models

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

## Scripts

| Script | Description |
|--------|-------------|
| `nhanes_physica_physics.py` | Shannon entropy, Markov, spectral gap, EPR |
| `nhanes_physica_ultimate.py` | EPR, time-varying Markov (Day/Night), PCA |
| `nhanes_threshold_robustness.py` | Q1/Q3 threshold robustness |
| `nhanes_logistic_validation.py` | Multivariate logistic (Age, Gender, Night P₀₁) |
| `nhanes_ultimate_logistic.py` | Logistic with BMI, PHQ-9 |
| `nhanes_strict_real_analysis.py` | Strict real-data pipeline (requires BMX_H, DPQ_H) |
