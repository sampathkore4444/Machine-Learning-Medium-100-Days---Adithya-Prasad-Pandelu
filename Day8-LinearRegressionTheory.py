import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Sample data
data = {
    "square_feet": [800, 1500, 2000, 1200, 1800],
    "num_rooms": [2, 3, 3, 2, 4],
    "house_price": [150000, 250000, 300000, 200000, 280000],
}
df = pd.DataFrame(data)

# Input and output variables
X = df[["square_feet", "num_rooms"]]
y = df["house_price"]

# Spliting train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model fitting
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print(f"R-squared: {r2:.2f}")
