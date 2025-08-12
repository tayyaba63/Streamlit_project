import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os 





# 1. Load dataset
df = pd.read_csv(r"E:\mohsineen python work\Microsoft VS Code\tayyaba\loan_prediction_project\data\loan_approval2_dataset.csv")
# 2. Drop ID column
df = df.drop(columns=['loan_id'])

# 3. Encode categorical columns
label_encoders = {}
categorical_cols = ['education',' self_employed',' loan_status']
 
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# 4. Separate features (X) and target (y)
X = df.drop(columns=[' loan_status'])
y = df[' loan_status']

# 5. Scale numeric columns manually
numeric_cols = [
    ' no_of_dependents', ' income_annum', ' loan_amount', ' loan_term',
    ' cibil_score', ' residential_assets_value', ' commercial_assets_value',
    ' luxury_assets_value', ' bank_asset_value'
]

scaler = StandardScaler()
X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

# 6. Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 7. Train model
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# 8. Predictions
y_pred = model.predict(X_test)

# 9. Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# 10. Ensure 'models' folder exists
os.makedirs("models", exist_ok=True)

# 11. Save model and preprocessing objects into 'models' folder
joblib.dump(model, "models/loan_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(label_encoders, "models/label_encoders.pkl")

# 12. Confirm saved files
print("\n✅ Model and preprocessing files saved in 'models' folder!")
print("Files in 'models' folder:", os.listdir("models"))