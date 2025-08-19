"""
Problem 1: Sales Prediction Using Simple Linear Regression
Dataset: advertising.csv

Objective: Our objective is to build a model that predicts sales based on advertising budgets allocated to
different media platforms. Here, we’ll focus on TV advertising to create a simple linear regression model.
Simple linear regression involves one predictor variable (in this case, TV spending) and aims to understand
how changes in this variable affect our target, Sales.
"""

# Import libraries
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Sample data
df = pd.read_csv("advertising.csv")
print(df)

# Features and target
X = df[["TV"]]
y = df["Sales"]

# Splitting into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Load the model
model = LinearRegression()
model.fit(X_train, y_train)

# predict the model
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean squared error:", mse)
print("R-squared error:", r2)

# Features and target
X = df[["TV"]]
y = df["Sales"]

# Splitting into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Load the model
model = LinearRegression()
model.fit(X_train, y_train)

# predict the model
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean squared error:", mse)
print("R-squared error:", r2)

# Visualizing the regression line
plt.title("TV Advertisement vs Sales")
plt.scatter(X, y, color="blue", label="Actual Sales")  # All data points
plt.plot(
    X_test, y_pred, color="red", linewidth=2, label="Regression Line"
)  # Regression line
plt.xlabel("TV Advertsing budget")
plt.ylabel("Sales")
plt.legend()
plt.show()


# how to draw the inference from the red line
"""
The **red regression line** in your scatter plot represents the **best-fit linear relationship** between **TV advertising budget (X-axis)** and **Sales (Y-axis)**. Here's how to interpret it:

---

### **Key Inferences from the Regression Line**
1. **Trend Direction**  
   - If the line slopes **upward** (positive slope), it means that as **TV ad spending increases, sales also tend to increase**.  
   - If it slopes **downward** (negative slope), higher ad spending is associated with lower sales (unlikely in this case).  
   - A **flat line** would mean TV ads have no effect on sales.

2. **Strength of the Relationship**  
   - If most of the **blue dots (actual data points)** are **close to the red line**, the model fits well, meaning TV ads strongly influence sales.  
   - If the dots are **widely scattered**, TV ads alone may not be a strong predictor of sales.

3. **Prediction Capability**  
   - The line allows you to **estimate sales** for any given TV ad budget.  
   - Example: If the line passes through `(TV=100, Sales=10)`, spending **$100K on TV ads** is expected to generate **$10K in sales**.

4. **Outliers & Anomalies**  
   - Points **far above the line** → Higher sales than expected for that ad spend.  
   - Points **far below the line** → Lower sales than expected.

---

### **How to Extract Numerical Insights**
Your code already calculates two key metrics:
```python
print("Mean squared error:", mse)  # Lower = Better fit
print("R-squared error:", r2)      # Closer to 1 = Stronger relationship
```
- **R-squared (R²)**:  
  - If `R² = 0.8`, it means **80% of sales variation** is explained by TV ad spending.  
  - If `R² < 0.5`, the relationship is weak, and other factors (e.g., radio, social media) may matter more.

- **Mean Squared Error (MSE)**:  
  - Measures average prediction error.  
  - Example: `MSE = 3` implies predictions are off by about `√3 ≈ 1.73` units (e.g., $1,730 in sales).

---

### **Practical Example**
Suppose your regression equation is:  
**`Sales = 7 + 0.05 × TV`**  
- **Interpretation**:  
  - **Intercept (7)**: If `TV = 0`, expected sales are **7 units** (baseline sales without ads).  
  - **Slope (0.05)**: For every **$1K increase in TV ads**, sales increase by **0.05 units** (e.g., $50 if units are in $1,000s).

---

### **Final Takeaways**
✅ **Upward slope?** → TV ads boost sales.  
✅ **R² close to 1?** → TV budget is a strong predictor.  
✅ **MSE low?** → Predictions are accurate.  
❌ **Dots scattered far from the line?** → Consider adding more features (e.g., radio, newspaper ads) to improve the model.

Would you like help refining the model further (e.g., adding more variables)?




what value of mse is lower to confirm it is a better fit?

The **Mean Squared Error (MSE)** measures the average squared difference between the **actual** and **predicted** values. A **lower MSE** indicates a **better fit**, but the interpretation depends on:  

1. **Scale of your target variable (`Sales`)**  
   - If `Sales` ranges from **0 to 50**, an MSE of **10** might be high.  
   - If `Sales` ranges from **0 to 1000**, an MSE of **10** is excellent.  

2. **Comparison with baseline models**  
   - Compare your model’s MSE to:  
     - A **simple mean-prediction model** (predicting average sales for all inputs).  
     - Other algorithms (e.g., decision trees, random forests).  

---

### **How to Judge if MSE is "Low Enough"?**
#### **Option 1: Compare to the Variance of `y`**
- Calculate the **variance of your target variable (`Sales`)**:
  ```python
  y_variance = y.var()
  print("Variance of Sales:", y_variance)
  ```
  - If **MSE << Variance**, your model is better than just guessing the mean.  
  - Example:  
    - Variance of `Sales` = **50**  
    - Your MSE = **10** → **Good fit** (much lower than 50).  

#### **Option 2: Relative MSE (Normalized Metrics)**
- Use **Root Mean Squared Error (RMSE = √MSE)** for interpretability in the original units.  
  ```python
  rmse = mean_squared_error(y_test, y_pred, squared=False)
  print("RMSE:", rmse)
  ```
  - If `RMSE = 3.16`, predictions are off by **±3.16 units** (e.g., $3,160 if `Sales` is in $1,000s).  
  - Compare to the **average sales**:
    ```python
    print("Mean Sales:", y.mean())
    ```
    - If mean sales = **15**, an RMSE of **3.16** (~21% error) may or may not be acceptable depending on business needs.

#### **Option 3: Rule of Thumb (Heuristic)**
- **MSE < 1** → Very good (for small-scale data).  
- **MSE between 1–10** → Moderate (common in many business cases).  
- **MSE > 10** → May need improvement (unless `y` has a large range).  

---

### **Example from Your Code**
Suppose your output was:
```
Mean squared error: 10.0
R-squared error: 0.75
```
- **Interpretation**:  
  - **MSE = 10** → If `Sales` ranges from **5 to 50**, this is decent.  
  - **R² = 0.75** → **75% of variance** is explained by TV ads (strong relationship).  
  - **Next Step**:  
    - If MSE is still too high, try adding more features (e.g., `Radio`, `Newspaper` ads).  
    - Compare to a baseline (e.g., `mean_squared_error(y_test, [y.mean()]*len(y_test))`).

---

### **Key Takeaway**
There’s **no universal "good" MSE**—it depends on your data scale and business context. Use:  
✅ **RMSE** for interpretability in original units.  
✅ **R²** to check explanatory power.  
✅ **Comparison tests** (e.g., vs. a mean-prediction model).  

Would you like help improving the model further?
"""
