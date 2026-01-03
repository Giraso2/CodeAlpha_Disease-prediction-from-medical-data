import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import urllib.request

# Create data folder if it doesn't exist
os.makedirs("data", exist_ok=True)

# Path to CSV
csv_path = "data/diabetes.csv"

# Download dataset if not present
if not os.path.exists(csv_path):
    print("Downloading dataset...")
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    # Column names for the dataset
    col_names = [
        "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
        "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"
    ]
    # Read directly from URL and save locally
    df = pd.read_csv(url, names=col_names)
    df.to_csv(csv_path, index=False)
    print("Dataset downloaded and saved to", csv_path)
else:
    df = pd.read_csv(csv_path)
    print("Dataset loaded from", csv_path)

# Separate features and target
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
