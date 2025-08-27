import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# data
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)


print(df.info())
print(df.describe())

# dropping columns that are not useful for predictions
df = df.drop(["PassengerId", "Name", "Ticket", "Cabin", "Embarked"], axis=1)
print(df.head())

# filling missing values for age with median
df["Age"].fillna(df["Age"].median(), inplace=True)
print(df.info())

# convert caregorical value to numerical value for Sex column
df["Sex"] = df["Sex"].map({"male": 1, "female": 0})
print(df.head())


# define features/predictors
X = df.drop("Survived", axis=1)
y = df["Survived"]
print(y.head())

# splitting the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# load the model and train it with training data
model = DecisionTreeClassifier(max_depth=4, random_state=42)
model.fit(X_train, y_train)

# predict on test set
y_pred = model.predict(X_test)

# evaluate the model
accuracy_score = accuracy_score(y_test, y_pred) * 100
print(f"Accuracy Score=, {accuracy_score:.2f}")

# visualize the decision tree
plt.figure(figsize=(20, 10))
tree.plot_tree(
    model,
    feature_names=X.columns,
    class_names=["Did not survive", "Survived"],
    filled=True,
)
plt.show()
