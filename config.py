"""Feature definitions for each disease model (used by train.py and app.py)."""

DISEASES = {
    "Diabetes": {
        "file": "data/diabetes.csv",
        "target": "Outcome",
        "positive_label": 1,
        "features": {
            "Pregnancies": ("Pregnancies", 0, 17, 1, 1),
            "Glucose": ("Glucose (mg/dL)", 0, 200, 120, 1),
            "BloodPressure": ("Blood Pressure (mm Hg)", 0, 125, 70, 1),
            "SkinThickness": ("Skin Thickness (mm)", 0, 99, 20, 1),
            "Insulin": ("Insulin (mu U/ml)", 0, 850, 80, 1),
            "BMI": ("BMI", 0.0, 70.0, 25.0, 0.1),
            "DiabetesPedigreeFunction": ("Diabetes Pedigree Function", 0.05, 2.5, 0.4, 0.01),
            "Age": ("Age (years)", 18, 90, 30, 1),
        },
        "advice_high": [
            "Consult a doctor / diabetologist for an HbA1c and fasting glucose test.",
            "Reduce refined sugar and white-flour foods; prefer whole grains and vegetables.",
            "Aim for at least 30 minutes of brisk walking daily.",
            "Maintain a healthy weight and monitor blood sugar regularly.",
        ],
        "advice_low": [
            "Keep a balanced diet and stay physically active.",
            "Get a routine blood sugar check once a year.",
        ],
    },
    "Heart Disease": {
        "file": "data/heart.csv",
        "target": "target",
        "positive_label": 1,
        "features": {
            "age": ("Age (years)", 20, 90, 50, 1),
            "sex": ("Sex (1 = Male, 0 = Female)", 0, 1, 1, 1),
            "cp": ("Chest Pain Type (0-3)", 0, 3, 0, 1),
            "trestbps": ("Resting Blood Pressure (mm Hg)", 80, 200, 120, 1),
            "chol": ("Cholesterol (mg/dL)", 100, 600, 200, 1),
            "fbs": ("Fasting Blood Sugar > 120 (1 = Yes, 0 = No)", 0, 1, 0, 1),
            "restecg": ("Resting ECG Result (0-2)", 0, 2, 0, 1),
            "thalach": ("Max Heart Rate Achieved", 60, 220, 150, 1),
            "exang": ("Exercise-induced Angina (1 = Yes, 0 = No)", 0, 1, 0, 1),
            "oldpeak": ("ST Depression (oldpeak)", 0.0, 7.0, 1.0, 0.1),
            "slope": ("Slope of Peak Exercise ST (0-2)", 0, 2, 1, 1),
            "ca": ("Major Vessels Coloured (0-4)", 0, 4, 0, 1),
            "thal": ("Thalassemia (0-3)", 0, 3, 2, 1),
        },
        "advice_high": [
            "Please see a cardiologist for an ECG / stress test.",
            "Cut down on salt, fried food and saturated fat.",
            "Avoid smoking and limit alcohol.",
            "Check blood pressure and cholesterol regularly.",
        ],
        "advice_low": [
            "Keep up regular exercise and a heart-healthy diet.",
            "Check blood pressure and cholesterol once a year.",
        ],
    },
    "Breast Cancer": {
        "file": None,  # loaded from scikit-learn's built-in dataset
        "target": "target",
        "positive_label": 0,  # in sklearn's dataset 0 = malignant
        "features": {
            "mean radius": ("Mean Radius", 6.0, 29.0, 14.0, 0.1),
            "mean texture": ("Mean Texture", 9.0, 40.0, 19.0, 0.1),
            "mean perimeter": ("Mean Perimeter", 43.0, 190.0, 92.0, 0.5),
            "mean area": ("Mean Area", 140.0, 2500.0, 650.0, 5.0),
            "mean smoothness": ("Mean Smoothness", 0.05, 0.17, 0.10, 0.001),
            "mean concavity": ("Mean Concavity", 0.0, 0.43, 0.09, 0.001),
        },
        "advice_high": [
            "Consult an oncologist / breast specialist for further tests (biopsy, imaging).",
            "Do not panic: this tool is a screening aid, not a diagnosis.",
        ],
        "advice_low": [
            "Continue regular self-examination and routine screening as advised by your doctor.",
        ],
    },
}
