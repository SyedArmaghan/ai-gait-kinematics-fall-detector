import numpy as np
import pandas as pd
from scipy.signal import find_peaks, butter, filtfilt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def butter_lowpass_filter(data, cutoff=4.0, fs=50.0, order=4):
    """Zero-phase Butterworth filter to suppress high-frequency movement noise."""
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

def extract_biomechanical_features(accel_mag, gyro_mag, fs=50.0):
    """
    Extracts clinical gait & kinematics metrics:
    - Cadence (steps/min)
    - Stride Time Mean & Coefficient of Variation (CV%)
    - Dynamic Stability via RMS acceleration
    - Peak Impact Acceleration
    """
    filtered_accel = butter_lowpass_filter(accel_mag, cutoff=4.0, fs=fs)
    
    # Peak detection for heel-strike gait cycles
    peaks, _ = find_peaks(filtered_accel, distance=fs * 0.4, prominence=0.35)
    
    stride_times = np.diff(peaks) / fs if len(peaks) >= 3 else np.array([1.0])
    
    cadence = (len(peaks) / (len(accel_mag) / fs)) * 60.0
    stride_mean = float(np.mean(stride_times))
    stride_std = float(np.std(stride_times))
    stride_cv = (stride_std / stride_mean) * 100.0 if stride_mean > 0 else 0.0  # Variability %
    
    rms_accel = float(np.sqrt(np.mean(filtered_accel ** 2)))
    peak_impact = float(np.max(filtered_accel))
    mean_rot_velocity = float(np.mean(gyro_mag))
    rot_std = float(np.std(gyro_mag))

    return {
        "cadence_spm": cadence,
        "stride_mean_s": stride_mean,
        "stride_variability_cv": stride_cv,
        "rms_accel": rms_accel,
        "peak_impact": peak_impact,
        "mean_rot_velocity": mean_rot_velocity,
        "rot_std": rot_std
    }

def generate_synthetic_kinematic_dataset(n_samples=300, fs=50.0, duration=10):
    """Generates realistic kinematics data for normal walking vs high fall risk."""
    np.random.seed(42)
    t = np.linspace(0, duration, int(fs * duration))
    data = []

    for i in range(n_samples):
        is_fall_risk = 1 if i >= n_samples // 2 else 0

        # Normal gait: steady frequency (~1.8 Hz) and low variability
        # Abnormal gait: lower cadence, irregular rhythm, higher variance
        freq = np.random.uniform(1.7, 2.1) if not is_fall_risk else np.random.uniform(1.0, 1.45)
        jitter = 0.05 if not is_fall_risk else np.random.uniform(0.18, 0.45)
        
        accel = 1.0 + 0.6 * np.sin(2 * np.pi * freq * t + np.random.uniform(0, 1)) + np.random.normal(0, jitter, len(t))
        gyro = 1.2 + 0.4 * np.cos(2 * np.pi * freq * t) + np.random.normal(0, jitter * 1.5, len(t))
        
        features = extract_biomechanical_features(accel, gyro, fs=fs)
        features["fall_risk_label"] = is_fall_risk
        data.append(features)

    return pd.DataFrame(data)

if __name__ == "__main__":
    print("Generating kinematic dataset and extracting biomechanical indicators...")
    df = generate_synthetic_kinematic_dataset(n_samples=300)
    
    X = df.drop(columns=["fall_risk_label"])
    y = df["fall_risk_label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    print("\nModel Evaluation - Fall-Risk Classification:")
    print(classification_report(y_test, preds, target_names=["Normal Gait", "High Fall Risk"]))