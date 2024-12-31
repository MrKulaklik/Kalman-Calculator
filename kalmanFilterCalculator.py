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

def kalman_filter(x,P,A,H,R,Q,B,u,z):
    x_pred = A @ x + B @ u # State Predicted
    P_pred = A @ P @ A.T + Q # Covariance Prediction

    I = np.eye(A.shape[0])
    K = P_pred @ H.T @ np.linalg.inv(H @ P_pred @ H.T + R) # Kalman Gain Calculation
    x_updated = x_pred + K @ (z - H @ x_pred) # State Update
    P_updated = (I - K @ H) @ P_pred # Covariance Update

    return x_updated, P_updated

def kalman_print(K):
    x_updated,P_updated = K
    print("Updated State Vector:")
    print(f"x: {x_updated[0]}")
    print(f"y: {x_updated[1]}")
    print(f"v_x: {x_updated[2]}")
    print(f"v_y: {x_updated[3]}")
    
    print("\nUpdated Covariance Matrix:")
    for i in range(4):
        for j in range(4):
            print(f"P[{i},{j}]: {P_updated[i, j]}")

kalman_print(kalman_filter(x,P,A,H,R,Q,B,u,z))