import numpy as np
import matplotlib.pyplot as plt

# Sample data: [Study Hours], [Pass (1) or Fail (0)]
study_hours = np.array([1, 2, 3, 4, 5, 6, 7])
passed = np.array([0, 0, 0, 1, 1, 1, 1])  # labels (y)


def sigmoid(z):
    # squashes input z to a range 0-1
    return 1 / (1 + np.exp(-z))


slope_m = 0
intercept_b = 0
learning_rate = 0.5
num_of_iterations = 100000
num_samples = len(passed)
cost_history = []

for i in range(num_of_iterations):
    # Step 1: Calculate Z (the linear part) and predictions (probabilities)
    z = slope_m * study_hours + intercept_b
    predictions = sigmoid(z)

    #  Step 2: Calculate the Cost (Log Loss) - for monitoring only
    # Avoid log(0) which is undefined by using np.clip
    predictions_clipped = np.clip(predictions, 1e-10, 1 - 1e-10)
    cost = (1 / num_samples) * (
        np.sum(
            passed * np.log(predictions_clipped)
            + (1 - passed) * np.log(1 - predictions_clipped)
        )
    )
    cost_history.append(cost)

    # Step 3: Calculate the Gradients (This is the core of the update!)
    # These formulas come from the derivative of the Log Loss cost function
    error = predictions - passed
    partial_derivate_b = (1 / num_samples) * np.sum(error)
    partial_derivate_m = (1 / num_samples) * np.sum(error * study_hours)

    # Step 4: Update the parameters
    intercept_b = intercept_b - learning_rate * partial_derivate_b
    slope_m = slope_m - learning_rate * partial_derivate_m

    # print progress for every 1000 iterations
    if i % 1000 == 0:
        print(f"Iteration {i}: Cost: {cost:.2f}")

print("=== Training Completed ===")
print(f"Optimized slope:, {slope_m:.4f}")
print(f"Optimized intercept:, {intercept_b:.4f}")

# Step 5: Make predictions with the trained model
print("\n--- Model Predictions (Probabilities) ---")
test_hours = np.array([2, 4.5, 7])
z_test = slope_m * test_hours + intercept_b
probabilities = sigmoid(z_test)

for hrs, prob in zip(test_hours, probabilities):
    print(
        f"Study {hrs} hrs -> Pass probability: {prob:.4f} | Class: {'Pass' if prob >= 0.5 else 'Fail'}"
    )

# Let's see the final decision boundary (where probability = 0.5)
# Solve for x: 0.5 = σ(m*x + b) --> m*x + b = 0
decision_boundary_x = -intercept_b / slope_m
print(
    f"\nDecision Boundary: Students who study more than {decision_boundary_x:.2f} hours are predicted to pass."
)
