import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# load data
df = pd.read_csv("TelCoChurn.csv")
# print(df.head())
# print(df.info())
# print(df.describe())

# data preprocessing
df["Churn"] = df["Churn"].map({False: 0, True: 1})

# print(df["International plan"].unique())
df["International plan"] = df["International plan"].map({"No": 0, "Yes": 1})

# print(df["Voice mail plan"].unique())
df["Voice mail plan"] = df["Voice mail plan"].map({"No": 0, "Yes": 1})

print(df["State"].unique())
df = pd.get_dummies(df, columns=["State"], drop_first=True)
print(df)

# Features/Predictors and Target
X = df.drop(columns="Churn", axis=1)
# print(X.head())
y = df["Churn"]
print(y.head())

# splitting into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Adaboost classifier - load the model and feed the training data to it
ada_model = AdaBoostClassifier(n_estimators=50, random_state=42)
ada_model.fit(X_train, y_train)

# Evaluate the Adaboost classifier model
ada_y_pred = ada_model.predict(X_test)
ada_accuracy_score = accuracy_score(y_test, ada_y_pred)
ada_classification_report = classification_report(y_test, ada_y_pred)
print(f"Adaboost accuracy score:, {ada_accuracy_score * 100:.2f}%")
print("Adaboost classification report:\n", ada_classification_report)

# Gradientboost classifier - load the model and feed the training data to it
gradient_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
gradient_model.fit(X_train, y_train)

# Evaluate the Gradientboost classifier model
gradient_y_pred = gradient_model.predict(X_test)
gradient_accuracy_score = accuracy_score(y_test, gradient_y_pred)
gradient_classification_report = classification_report(y_test, gradient_y_pred)
print(f"Gradientboost accuracy score:, {gradient_accuracy_score * 100:.2f}%")
print("Gradientboost classification report:\n", gradient_classification_report)
