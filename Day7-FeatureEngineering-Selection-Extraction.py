import pandas as pd

# sample data
data = {
    "square_footage": [1500, 1800, 2400, 3000, 3500],
    "num_of_bedrooms": [3, 4, 3, 5, 4],
    "num_of_bathrooms": [2, 2, 3, 3, 4],
    "location_rating": [8, 7, 9, 6, 7],
    "price": [300000, 350000, 450000, 500000, 600000],  # Target
}
df = pd.DataFrame(data)
print(df)

# Do Feature engineering
data = df[
    [
        "square_footage",
        "num_of_bedrooms",
        "num_of_bathrooms",
        "location_rating",
        "price",
    ]
]
correlation_matrix = data.corr()
print(correlation_matrix)  # dataframe
print(correlation_matrix["price"].sort_values(ascending=False))

# same as above
# corr_with_target = df.corr()["price"].sort_values(ascending=False)
# print("Correlation with Price:\n", corr_with_target)

# Feature Selection: Select 'location_rating' and 'square_footage' based on correlation analysis
# we selected "location_rating" despite having a negative correlation
""""
(higher ratings → lower prices), which is unusual for real estate. This could happen if:
The "higher" ratings actually represent worse locations (e.g., 10 = risky area, 1 = premium area).
Small sample size (only 5 points) causing noise.
"""
selected_features = df[["location_rating", "square_footage"]]


# Forward Selection
# Iteratively add the best-performing feature:
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SequentialFeatureSelector

# features and target
X = df[["square_footage", "num_of_bedrooms", "num_of_bathrooms", "location_rating"]]
y = df["price"]
lr = LinearRegression()
sfs = SequentialFeatureSelector(lr, n_features_to_select=2, direction="forward")
sfs.fit(X, y)
# Get the selected features
selected_features = sfs.get_support()
print("Selected Features (Forward):", selected_features)
print("Selected Features (Forward):", list(X.columns[selected_features]))

# Backward Elimination
# Iteratively remove the worst-performing feature:
sbs = SequentialFeatureSelector(lr, n_features_to_select=2, direction="backward")
sbs.fit(X, y)
# Get the selected features
selected_features = sbs.get_support()
print("Selected Features (Backward):", selected_features)
print("Selected Features (Backward):", list(X.columns[selected_features]))


# Feature Extraction: Create new feature 'rooms_per_sqft'
df["rooms_per_sqft"] = df["square_footage"] / (
    df["num_of_bedrooms"] + df["num_of_bathrooms"]
)

"""
Polynomial features are used in machine learning to introduce non-linearity into models, 
allowing them to capture more complex relationships in the data. 
They are particularly useful when linear models struggle to fit data with curved or intricate patterns. 
This technique involves creating new features by raising existing ones to powers (e.g., squaring or cubing) 
and also by combining them (e.g., multiplying two features together). 
"""

# Add a Polynomial Feature for "square_footage"
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2, include_bias=False)
square_footage_poly = poly.fit_transform(df[["square_footage"]])
print(square_footage_poly)
print(square_footage_poly[:, 1])

"""
his creates:

Original feature: square_footage (degree 1)

Squared feature: square_footage² (degree 2)

Why Use Polynomial Features?
Non-linear Relationships: Real-world relationships often aren't straight lines

Better Fit: Can model curves in your data

Interaction Effects: Can capture how features combine to affect the target

Example Visualization:
Imagine plotting square_footage vs price:

Linear relationship: Straight line

Polynomial (degree 2): Parabola that can curve to fit data better

Potential Issues:
Overfitting: High-degree polynomials can fit noise in your data

Feature Explosion: Number of features grows rapidly with degree and original features

Practical Tip:
Start with degree 2 or 3 and use regularization to prevent overfitting. Always visualize the relationship first to see if polynomial features might help.
"""

"""
Here’s a **step-by-step visualization guide** to understand polynomial relationships and prevent overfitting, using your housing data:

---

### 1. **Visualizing Non-Linear Relationships**
We’ll use `square_footage` (strongest predictor) vs `price` to see if polynomial features help:

```python
# Scatter plot with polynomial trendline
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='square_footage', y='price', s=100)

# Fit and plot polynomial trends
sns.regplot(data=df, x='square_footage', y='price', 
            order=1, scatter=False, label='Linear (degree=1)', color='blue')
sns.regplot(data=df, x='square_footage', y='price', 
            order=2, scatter=False, label='Quadratic (degree=2)', color='red')
sns.regplot(data=df, x='square_footage', y='price', 
            order=3, scatter=False, label='Cubic (degree=3)', color='green')

plt.title("Polynomial Fit Comparison")
plt.legend()
plt.show()
```

**Key Observations**:
- If the red/green curves fit the dots better than the blue line, polynomials help.
- If high-degree curves start twisting oddly between points, that's overfitting.

---

### 2. **Regularization to Prevent Overfitting**
When using polynomial features in models, add regularization:

```python
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

# Create polynomial features (degree=3)
poly = PolynomialFeatures(degree=3, include_bias=False)
X_poly = poly.fit_transform(df[['square_footage']])

# Standardize features (critical for regularization)
scaler = StandardScaler()
X_poly_scaled = scaler.fit_transform(X_poly)

# Fit with regularization
ridge = Ridge(alpha=1.0)  # alpha controls regularization strength
ridge.fit(X_poly_scaled, df['price'])

# Check coefficients
print("Polynomial coefficients:", ridge.coef_)
```

**Output Interpretation**:
- Small, balanced coefficients = good regularization.
- Huge coefficients = overfitting (increase `alpha`).

---

### 3. **Practical Validation**
Check if polynomials actually improve predictions using cross-validation:

```python
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline

# Linear model
linear_score = cross_val_score(make_pipeline(StandardScaler(), Ridge()), 
                             df[['square_footage']], df['price'], cv=3).mean()

# Polynomial model
poly_score = cross_val_score(
    make_pipeline(
        PolynomialFeatures(degree=2),
        StandardScaler(),
        Ridge(alpha=1.0)
    ),
    df[['square_footage']], df['price'], cv=3).mean()

print(f"Linear R²: {linear_score:.3f}, Polynomial R²: {poly_score:.3f}")
```

**Decision Rule**:
- If polynomial score > linear score → Use polynomials.
- If scores are close → Simpler model (linear) is better.

---

### 4. **Advanced Visualization (3D for Interactions)**
For interactions between two features (e.g., `square_footage` × `location_rating`):

```python
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df['square_footage'], df['location_rating'], df['price'], c='r', marker='o', s=100)
ax.set_xlabel('Square Footage')
ax.set_ylabel('Location Rating')
ax.set_zlabel('Price')
plt.title("3D Feature Interaction")
plt.show()
```

**Pro Tip**:
- If points form a curved surface (not flat), polynomial interactions may help.

---

### Key Takeaways:
1. **Start Simple**: Always try linear models first.
2. **Visualize First**: Use scatter/3D plots to spot non-linear patterns.
3. **Regularize**: Always scale features and use Ridge/Lasso with polynomials.
4. **Validate**: Cross-check if polynomials actually improve performance.

Want me to focus on any specific part (e.g., tuning regularization strength)?
"""
