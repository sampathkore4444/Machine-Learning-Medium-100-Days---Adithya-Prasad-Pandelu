import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv("employee_satisfaction.csv")
print(df)

# Overview of data structure
df.info()
print(df.describe())

# Univariate analysis

# Plot age distribution
plt.title("Age Distribution")
plt.xlabel("Age of a person")
sns.histplot(df["age"], bins=10, kde=True)
plt.show()

# Plot satisfaction ratings
plt.title("Employee Satisfaction Distribution")
plt.xlabel("Employee satisfaction")
sns.histplot(df["satisfaction_rating"], bins=10, kde=True)
plt.show()

# Bivariate analysis
# check if income satisfies employees
x = df["monthly_income"]
y = df["satisfaction_rating"]
plt.title("Income vs Employee Satisfaction Distribution")
plt.xlabel("Monthly Income")
plt.ylabel("Employee satisfaction rating")
sns.scatterplot(x="monthly_income", y="satisfaction_rating", data=df)
plt.show()

# Group analysis
# Mean satisfaction by department
avg_satisfaction_by_dept = df.groupby("department")["satisfaction_rating"].mean()
print("Average satisfaction Rating by Department:\n", avg_satisfaction_by_dept)

# Mean satisfaction by job role
avg_satisfaction_by_role = df.groupby("job_role")["satisfaction_rating"].mean()
print("Average satisfaction by job role:\n", avg_satisfaction_by_role)

# Multivariate analysis
data = df[["age", "years_at_company", "satisfaction_rating", "monthly_income"]]
correlation_matrix = data.corr()
plt.title("Correlation Matrix of employee data")
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.show()
