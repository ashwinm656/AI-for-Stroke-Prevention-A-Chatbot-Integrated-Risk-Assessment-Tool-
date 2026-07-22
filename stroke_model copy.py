import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pickle

# Load dataset
df = pd.read_csv('stroke_data.csv')

# Encode gender
df['gender'] = LabelEncoder().fit_transform(df['gender'])

# Define input features and target
X = df.drop(['stroke_risk_percentage', 'at_risk'], axis=1)
y = df['at_risk']

# Normalize age
scaler = StandardScaler()
X['age'] = scaler.fit_transform(X[['age']])

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
# n_estimators/max_depth capped so the pickled model stays a reasonable
# size to push to GitHub / deploy on Streamlit Cloud (unbounded trees on
# 35k rows produced an 80MB+ file).
model = RandomForestClassifier(
    n_estimators=150, max_depth=10, min_samples_leaf=5,
    class_weight='balanced', random_state=42,
)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Save model and scaler
pickle.dump(model, open('stroke_model.pkl', 'wb'))
pickle.dump(scaler, open('scaler.pkl', 'wb'))
