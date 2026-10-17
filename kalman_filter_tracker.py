import numpy as np
import matplotlib.pyplot as plt

def run_kalman_tracker():
    """
    2D Airborne Target State Estimation using Discrete Linear Kalman Filter.
    Applies numerical state-space models to filter radar measurement noise.
    """
    dt = 0.5
    steps = 40
    
    # State Transition Matrix (Constant Velocity Model)
    F = np.array([[1, dt, 0, 0],
                  [0, 1,  0, 0],
                  [0, 0,  1, dt],
                  [0, 0,  0, 1]])
    
    # Observation Matrix (Position only)
    H = np.array([[1, 0, 0, 0],
                  [0, 0, 1, 0]])
    
    Q = np.eye(4) * 0.1   # Process Noise Covariance
    R = np.eye(2) * 15.0  # Measurement Noise Covariance
    P = np.eye(4) * 5.0   # Initial Estimate Covariance
    
    # Ground Truth Trajectory
    true_x = np.linspace(0, 400, steps)
    true_y = 200 + 100 * np.sin(np.linspace(0, 3, steps))
    
    measurements = np.column_stack([
        true_x + np.random.normal(0, np.sqrt(R[0, 0]), steps),
        true_y + np.random.normal(0, np.sqrt(R[1, 1]), steps)
    ])
    
    estimated_states = []
    x = np.array([[0], [10], [200], [2]])  # Initial state
    
    for z in measurements:
        # Prediction Step
        x = np.dot(F, x)
        P = np.dot(np.dot(F, P), F.T) + Q
        
        # Update Step
        z = z.reshape(2, 1)
        y = z - np.dot(H, x)
        S = np.dot(np.dot(H, P), H.T) + R
        K = np.dot(np.dot(P, H.T), np.linalg.inv(S))
        x = x + np.dot(K, y)
        P = P - np.dot(np.dot(K, H), P)
        
        estimated_states.append(x.flatten())
        
    est = np.array(estimated_states)
    
    # Plot Trajectory
    plt.figure(figsize=(10, 6))
    plt.plot(true_x, true_y, 'g--', label="Ground Truth Trajectory", linewidth=2)
    plt.scatter(measurements[:, 0], measurements[:, 1], color='red', alpha=0.6, label="Noisy Radar Measurements")
    plt.plot(est[:, 0], est[:, 2], 'b-', label="Kalman Filter Estimate", linewidth=2)
    plt.title("2D Airborne Target State Estimation (Kalman Filtering)")
    plt.xlabel("X Position (meters)")
    plt.ylabel("Y Position (meters)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("kalman_tracking_result.png", dpi=300)
    print("Grafik 'kalman_tracking_result.png' olarak kaydedildi.")

if __name__ == "__main__":
    run_kalman_tracker()
