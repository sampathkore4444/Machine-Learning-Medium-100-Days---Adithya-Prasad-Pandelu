# Import required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv("TelCoChurn.csv")

# Data preprocessing
df["Churn"] = df["Churn"].astype(int)
df["International plan"] = df["International plan"].map({"Yes": 1, "No": 0})
df["Voice mail plan"] = df["Voice mail plan"].map({"Yes": 1, "No": 0})
df = pd.get_dummies(df, columns=["State"], drop_first=True)

X = df.drop(columns=["Churn"])  # Features
y = df["Churn"]  # Target variable

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
# AdaBoost Classifier
ada_model = AdaBoostClassifier(n_estimators=50, random_state=42)
ada_model.fit(X_train, y_train)
ada_preds = ada_model.predict(X_test)

# Evaluation
print("AdaBoost Accuracy:", accuracy_score(y_test, ada_preds))
print("AdaBoost Classification Report:\n", classification_report(y_test, ada_preds))
# Gradient Boosting Classifier
gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb_model.fit(X_train, y_train)
gb_preds = gb_model.predict(X_test)

# Evaluation
print("Gradient Boosting Accuracy:", accuracy_score(y_test, gb_preds))
print(
    "Gradient Boosting Classification Report:\n",
    classification_report(y_test, gb_preds),
)
