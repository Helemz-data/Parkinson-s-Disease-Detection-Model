import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("parkinsons.csv")

# Drop non-ML column
data = data.drop(columns=["name"])

X = data.drop(columns=["status"])
y = data["status"]

# Save feature names
feature_names = X.columns.tolist()
joblib.dump(feature_names, "feature_names.pkl")

# Scale data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# SVM Model
svm = SVC(kernel="rbf", probability=True)
svm.fit(X_train, y_train)
svm_acc = accuracy_score(y_test, svm.predict(X_test))

# Random Forest Model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_acc = accuracy_score(y_test, rf.predict(X_test))

print("SVM Accuracy:", svm_acc)
print("RF Accuracy:", rf_acc)

# Choose best model
best_model = svm if svm_acc >= rf_acc else rf

# Save model and scaler
joblib.dump(best_model, "parkinsons_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Training completed. Model saved.")

joblib.dump(svm, "svm_model.pkl")
joblib.dump(rf, "rf_model.pkl")
