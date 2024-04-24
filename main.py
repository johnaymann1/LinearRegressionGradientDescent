# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt

# Function to normalize features
def normalize_features(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    normalized_X = (X - mean) / std
    return normalized_X, mean, std

# Function to split data into training, validation, and test sets
def train_validate_test_split(data, labels, test_ratio=0.3, val_ratio=0.3):
    num_samples = len(data)
    num_test_samples = int(num_samples * test_ratio)
    num_val_samples = int(num_samples * val_ratio)
    num_train_samples = num_samples - num_test_samples - num_val_samples

    # Shuffle indices
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    data = data[indices]
    labels = labels[indices]

    # Split data
    X_train = data[:num_train_samples]
    y_train = labels[:num_train_samples]

    X_val = data[num_train_samples:num_train_samples + num_val_samples]
    y_val = labels[num_train_samples:num_train_samples + num_val_samples]

    X_test = data[num_train_samples + num_val_samples:]
    y_test = labels[num_train_samples + num_val_samples:]

    # Normalize features
    X_train_normalized, mean, std = normalize_features(X_train)
    X_val_normalized = (X_val - mean) / std
    X_test_normalized = (X_test - mean) / std

    return X_train_normalized, y_train, X_val_normalized, y_val, X_test_normalized, y_test

# Function to calculate the loss
def loss_function(X, w, Y):
    loss = 0.5 * np.dot((np.dot(X, w) - Y).T, (np.dot(X, w) - Y))
    return loss

# Function for gradient descent optimization
def gradient_descent(X, Y, weights, LR, iterations):
    for _ in range(iterations):
        weights = weights - (LR * np.dot(X.T, (np.dot(X, weights) - Y)))
    return weights

# Function to calculate R-squared
def calculate_r_squared(actual, predicted):
    mean_actual = np.mean(actual)
    ss_total = np.sum((actual - mean_actual) ** 2)
    ss_residual = np.sum((actual - predicted) ** 2)
    r_squared = 1 - (ss_residual / ss_total) if ss_total != 0 else 0
    return r_squared

# Main function
def main():
    # Step 1: Generate random data without a fixed seed
    W = np.array([6, 5, 3, 1.5])
    X1 = np.random.rand(1000)
    X2 = np.random.rand(1000)
    X3 = np.random.rand(1000)
    Y = W[0] + W[1] * X1 + W[2] * X2 + W[3] * X3
    
    dataset = np.vstack([X1, X2, X3]).T


    # Step 2: Split data using the provided function
    X_train, Y_train, X_val, Y_val, X_test, Y_test = train_validate_test_split(dataset, Y)

    # Step 3: Initialize weights and perform gradient descent (training)
    X_train_bias = np.hstack([np.ones((len(X_train), 1)), X_train])
    LR = 0.0001  # Learning rate
    n_iter = 1000  # Number of iterations
    final_weights = gradient_descent(X_train_bias, Y_train, W, LR, n_iter)

    # Step 4: Calculate and print accuracy (R^2) on the test set
    X_test_bias = np.hstack([np.ones((len(X_test), 1)), X_test])
    y_pred_test = np.dot(X_test_bias, final_weights)
    r_squared = calculate_r_squared(Y_test, y_pred_test)
    print(f"R-squared on Test Set: {r_squared}")
    print(f"Final weights: {final_weights}")

    # Plot using final weights
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.scatter(X_test[:, 0], Y_test, color='blue', label='Samples')
    x_line_final = np.linspace(np.min(X_test[:, 0]), np.max(X_test[:, 0]), 100)
    y_line_final = final_weights[0] + final_weights[1] * x_line_final + final_weights[2] * x_line_final + final_weights[3] * x_line_final
    plt.plot(x_line_final, y_line_final, color='red', label='Line (Final Weights)')
    plt.xlabel('X1')
    plt.ylabel('Y')
    plt.legend()
    plt.title('Linear Regression with Final Weights')

    # Set the same scale for both x-axis and y-axis
    plt.axis('equal')
    plt.xlim([np.min(X_test[:, 0]), np.max(X_test[:, 0])])
    plt.ylim([np.min(Y_test), np.max(Y_test)])

    # Plot using initial weights
    plt.subplot(1, 2, 2)
    plt.scatter(X_test[:, 0], Y_test, color='blue', label='Samples')
    x_line_initial = np.linspace(np.min(X_test[:, 0]), np.max(X_test[:, 0]), 100)
    y_line_initial = W[0] + W[1] * x_line_initial + W[2] * x_line_initial + W[3] * x_line_initial
    plt.plot(x_line_initial, y_line_initial, color='green', label='Line (Initial Weights)')
    plt.xlabel('X1')
    plt.ylabel('Y')
    plt.legend()
    plt.title('Linear Regression with Initial Weights')

    # Set the same scale for both x-axis and y-axis
    plt.axis('equal')
    plt.xlim([np.min(X_test[:, 0]), np.max(X_test[:, 0])])
    plt.ylim([np.min(Y_test), np.max(Y_test)])

    plt.show()

# Run the main function if the script is executed
if __name__ == "__main__":
    main()
