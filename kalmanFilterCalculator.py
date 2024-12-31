import numpy as np

# Define variables
x = np.array([0, 0, 1, 1])  # Initial state: x, y, v_x, v_y
P = np.array([[1, 0, 0, 0],  # Initial covariance matrix
              [0, 1, 0, 0],
              [0, 0, 1000, 0],
              [0, 0, 0, 1000]])
delta_t = 0.1  # Time step (100 ms)
A = np.array([
    [1, 0, delta_t, 0],
    [0, 1, 0, delta_t],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
])
B = np.array([
    [0.5 * delta_t**2, 0],
    [0, 0.5 * delta_t**2],
    [delta_t, 0],
    [0, delta_t]
])
H = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0]
])
R = np.diag([0.25, 0.25])  # Measurement noise covariance
Q = np.diag([0.01, 0.01, 0.1, 0.1])  # Process noise covariance
u = np.array([0.2, 0.2])  # Control input (acceleration)
z = np.array([0.95, 1.05])  # Measured positions (with noise)

# Prediction step
x_pred = A @ x + B @ u
P_pred = A @ P @ A.T + Q

# Update step
y = z - H @ x_pred  # Innovation
S = H @ P_pred @ H.T + R  # Innovation covariance
K = P_pred @ H.T @ np.linalg.inv(S) # Kalman gain
x_updated = x_pred + K @ y
P_updated = (np.eye(4) - K @ H) @ P_pred

print("Predicted State (x_pred):\n", x_pred)
print("\nPredicted Covariance (P_pred):\n", P_pred)
print("\nUpdated State (x_updated):\n", x_updated)
print("\nUpdated Covariance (P_updated):\n", P_updated)