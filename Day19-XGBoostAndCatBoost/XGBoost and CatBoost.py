import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
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
XG_model = XGBClassifier(n_estimators=100, max_depth=4, random_state=42)
XG_model.fit(X_train, y_train)

# Evaluate the Adaboost classifier model
XG_y_pred = XG_model.predict(X_test)
XG_accuracy_score = accuracy_score(y_test, XG_y_pred)
XG_classification_report = classification_report(y_test, XG_y_pred)
print(f"XGBoost accuracy score:, {XG_accuracy_score * 100:.2f}%")
print("XGBoost classification report:\n", XG_classification_report)

# Gradientboost classifier - load the model and feed the training data to it
cat_model = CatBoostClassifier(
    iterations=100, depth=4, learning_rate=0.1, verbose=False
)
cat_model.fit(X_train, y_train)

# Evaluate the Gradientboost classifier model
cat_y_pred = cat_model.predict(X_test)
cat_accuracy_score = accuracy_score(y_test, cat_y_pred)
cat_classification_report = classification_report(y_test, cat_y_pred)
print(f"CatBoost accuracy score:, {cat_accuracy_score * 100:.2f}%")
print("CatBoost classification report:\n", cat_classification_report)
