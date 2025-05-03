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
model = RandomForestClassifier(class_weight='balanced')
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Save model and scaler
pickle.dump(model, open('stroke_model.pkl', 'wb'))
pickle.dump(scaler, open('scaler.pkl', 'wb'))
