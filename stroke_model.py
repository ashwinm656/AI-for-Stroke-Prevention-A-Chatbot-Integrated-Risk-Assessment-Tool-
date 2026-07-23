"""
Trains the stroke risk model on the real "Stroke Risk Prediction Dataset
Based on Symptoms" (Kaggle, mahatiratusher/stroke-risk-prediction-dataset).

Expects stroke_risk_dataset.csv in this folder with these exact columns:
Chest Pain, Shortness of Breath, Irregular Heartbeat, Fatigue & Weakness,
Dizziness, Swelling (Edema), Pain in Neck/Jaw/Shoulder/Back,
Excessive Sweating, Persistent Cough, Nausea/Vomiting, High Blood Pressure,
Chest Discomfort (Activity), Cold Hands/Feet, Snoring/Sleep Apnea,
Anxiety/Feeling of Doom, Age, Stroke Risk (%), At Risk (Binary)

Note: this dataset has no gender column, so the model is trained on age +
the 15 symptoms only. app.py still asks for gender (useful to record) but
does not feed it into the model, since it was never trained on it.
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pickle

# Load dataset
df = pd.read_csv('stroke_risk_dataset.csv')
df = df.drop_duplicates()

# Column order here MUST match SYMPTOMS in app.py exactly (age scaled,
# then the 15 symptoms in this same order) — run_prediction() in app.py
# builds its input vector in this order.
FEATURE_COLUMNS = [
    'Age', 'Chest Pain', 'Shortness of Breath', 'Irregular Heartbeat',
    'Fatigue & Weakness', 'Dizziness', 'Swelling (Edema)',
    'Pain in Neck/Jaw/Shoulder/Back', 'Excessive Sweating', 'Persistent Cough',
    'Nausea/Vomiting', 'High Blood Pressure', 'Chest Discomfort (Activity)',
    'Cold Hands/Feet', 'Snoring/Sleep Apnea', 'Anxiety/Feeling of Doom',
]

X = df[FEATURE_COLUMNS].copy()
y = df['At Risk (Binary)']

# Normalize age (fit on age only, same as before)
scaler = StandardScaler()
X['Age'] = scaler.fit_transform(X[['Age']])

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y,
)

# Train model — capped size so the pickle stays reasonable for GitHub/
# Streamlit Cloud (unbounded trees on 70k rows get huge).
model = RandomForestClassifier(
    n_estimators=150, max_depth=10, min_samples_leaf=8,
    class_weight='balanced', random_state=42,
)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Save model and scaler
pickle.dump(model, open('stroke_model.pkl', 'wb'))
pickle.dump(scaler, open('scaler.pkl', 'wb'))
