# Wine Quality Prediction System

Section B classroom prototype using Python, scikit-learn, pandas, joblib, Streamlit, and the UCI Red Wine Quality dataset.

## Problem
A wine producer needs a consistent preliminary way to screen wine samples using recorded physicochemical measurements.

## Dataset
UCI Machine Learning Repository — Wine Quality:
https://archive.ics.uci.edu/dataset/186/wine+quality

The training script downloads the red-wine CSV automatically.

## Prediction rule
- Original quality score >= 6: `GOOD QUALITY` (class 1)
- Original quality score < 6: `NEEDS IMPROVEMENT` (class 0)

The original `quality` column is used only to create the target. It is removed from the predictor features to prevent target leakage.

## Input features
Fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free sulfur dioxide, total sulfur dioxide, density, pH, sulphates, and alcohol.

## Technologies
- Python
- pandas
- scikit-learn Random Forest
- joblib
- Streamlit
- GitHub

## Installation and run
Use Python 3.10 or later. Open a terminal in this folder and run:

```bash
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

The training script creates `wine_quality.csv` and `wine_model.pkl`. Open the local URL shown in the terminal, normally http://localhost:8501.

## Using the app
1. Open the Streamlit app.
2. Download the sample input CSV.
3. Upload a CSV containing the same 11 predictor columns.
4. Review the predicted category.
5. Download the prediction results CSV.

## Limitations
This is a classroom prototype trained on a particular public red-wine dataset. Results may not generalize to other wine types, producers, regions, or testing conditions. Predictions are preliminary screening aids and are not final quality-control decisions.
