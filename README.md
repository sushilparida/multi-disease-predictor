# AI-Powered Multi-Disease Prediction System

Predicts risk of **Diabetes, Heart Disease and Breast Cancer** with a separate ML model per disease.
Built with Python, scikit-learn and Streamlit.

## Setup (5 minutes)
1. `pip install -r requirements.txt`
2. Download two CSVs from Kaggle and put them in the `data/` folder:
   - `data/diabetes.csv`  -> "Pima Indians Diabetes Database" (columns: Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age, Outcome)
   - `data/heart.csv`     -> "Heart Disease Dataset" (columns: age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, target)
   - Breast cancer needs no download (built into scikit-learn).
3. `python train.py`   (trains and saves the models, prints accuracy / F1 / ROC-AUC)
4. `streamlit run app.py`

## Deploy (optional)
Push to GitHub -> share.streamlit.io -> select app.py -> public URL.

## Features
- Separate model per disease (Logistic Regression vs Random Forest, best chosen by ROC-AUC)
- Risk %, risk level (Low / Moderate / High)
- Top feature importances
- Lifestyle recommendations
- Downloadable PDF health report
- Model performance page
