"""
Problem 2: Student Performance Prediction Using Multiple Linear Regression

Dataset: Student_Performance.csv

Objective: Predict Performance Index based on multiple factors like Hours Studied, Previous Scores, Extracurricular Activities, Sleep Hours, and Sample Question Papers Practiced.
"""

# Import libraries
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# load data
df = pd.read_csv("Student_Performance.csv")
print(df)
print(df.info())
print(df.describe())

# Feature Engineering
df["Extracurricular_Activities"] = df["Extracurricular Activities"].apply(
    lambda x: 1 if x == "Yes" else 0
)
# drop the original column
df = df.drop(columns=["Extracurricular Activities"], axis=1)

# to see the correlation
data = df[
    [
        "Hours Studied",
        "Previous Scores",
        "Extracurricular_Activities",
        "Sleep Hours",
        "Sample Question Papers Practiced",
        "Performance Index",
    ]
]
correlation_matrix = data.corr()
plt.title("Correlation Matrix of employee data")
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.show()

# features and target
X = df[
    [
        "Hours Studied",
        "Previous Scores",
        "Extracurricular_Activities",
        "Sleep Hours",
        "Sample Question Papers Practiced",
    ]
]
y = df["Performance Index"]

# splitting into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# load and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions for test data
y_pred = model.predict(X_test)
# print(y_pred)

# evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean squared error:", mse)
print("R squared:", r2)
print(y_test.min(), y_test.max())

# Visualizing the predicted values against actual values
plt.title("Actual Performance Index vs Predicted Performance Index")
sns.scatterplot(x=y_test, y=y_pred, color="blue", alpha=0.7)
plt.plot(
    [y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--"
)  # reference line

plt.xlabel("Actual Performance Index")
plt.ylabel("Predicted Performance Index")
plt.show()
