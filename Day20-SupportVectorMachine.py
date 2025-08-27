import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

# data
digits = load_digits()
# print(digits)

# Features and Target
X = digits.data
y = digits.target

# Splitting into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# load the model and fit the training data to it
model = SVC(kernel="poly", gamma=0.01, C=100)
model.fit(X_train, y_train)

# predict on test data
y_pred = model.predict(X_test)

# evaluate the model
accuracy_score = accuracy_score(y_test, y_pred)
classification_report = classification_report(y_test, y_pred)

print(f"Accuracy Score= {accuracy_score * 100:.2f}")
print(f"Classification report=\n, {classification_report}")

# Visualize predictions
for index, (image, label) in enumerate(zip(X_test[:8], y_pred[:8])):
    plt.subplot(2, 4, index + 1)
    plt.imshow(image.reshape(8, 8), cmap="gray")
    plt.title(f"Predicted=, {label}")
plt.show()
# from sklearn.datasets import load_digits

# digits = load_digits()
# print(digits.images[0])

# import matplotlib.pyplot as plt

# plt.matshow(digits.images[0], cmap="gray")
# plt.matshow(digits.images[1], cmap="gray")

# plt.show()


# Hyperparameter Optimization using GridSearch CV
# import GridSearchCV
from sklearn.model_selection import GridSearchCV


# import SVC classifier
from sklearn.svm import SVC


# instantiate classifier with default hyperparameters with kernel=rbf, C=1.0 and gamma=auto
svc = SVC()


# declare parameters for hyperparameter tuning
parameters = [
    {"C": [1, 10, 100, 1000], "kernel": ["linear"]},
    {
        "C": [1, 10, 100, 1000],
        "kernel": ["rbf"],
        "gamma": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    },
    {
        "C": [1, 10, 100, 1000],
        "kernel": ["poly"],
        "degree": [2, 3, 4],
        "gamma": [0.01, 0.02, 0.03, 0.04, 0.05],
    },
]


grid_search = GridSearchCV(
    estimator=svc, param_grid=parameters, scoring="accuracy", cv=5, verbose=0
)


grid_search.fit(X_train, y_train)

# examine the best model


# best score achieved during the GridSearchCV
print("GridSearch CV best score : {:.4f}\n\n".format(grid_search.best_score_))


# print parameters that give the best results
print("Parameters that give the best results :", "\n\n", (grid_search.best_params_))


# print estimator that was chosen by the GridSearch
print(
    "\n\nEstimator that was chosen by the search :",
    "\n\n",
    (grid_search.best_estimator_),
)
print(
    "GridSearch CV score on test set: {0:0.4f}".format(
        grid_search.score(X_test, y_test)
    )
)
