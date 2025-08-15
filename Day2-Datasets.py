import pandas as pd

# sample dataset
data = {
    "StudentID": [1, 2, 3, 3, 4],
    "Grade": [85, None, 95, 95, 78],
    "Attendance": [95, 85, 88, 88, None],
    "Extracurricular": ["Football", "Debate", None, None, "Debate"],
}

df = pd.DataFrame(data)
print("Original Data:")
print(df)

# drop duplicates
df = df.drop_duplicates()
print("Data after removing duplicates:")
print(df)

# handling missing values
df["Grade"] = df["Grade"].fillna(df["Grade"].mean())  # Replace with mean grade
df["Attendance"] = df["Attendance"].fillna(
    df["Attendance"].median()
)  # Replace with median attendance
df["Extracurricular"] = df["Extracurricular"].fillna("No Activity")
print("Data after handling missing values:")
print(df)
