import pandas as pd
import numpy as np
from scipy.signal import find_peaks, butter, filtfilt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import classification_report, confusion_matrix

def butter_lowpass_filter(data, cutoff=4.0, fs=50.0, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

def extract_features_from_trial(trial_df, fs=50.0):
    # Calculate resultant 3D vector magnitudes
    accel_mag = np.sqrt(trial_df["accel_x"]**2 + trial_df["accel_y"]**2 + trial_df["accel_z"]**2).values
    gyro_mag = np.sqrt(trial_df["gyro_roll"]**2 + trial_df["gyro_pitch"]**2 + trial_df["gyro_yaw"]**2).values

    # Filter signal
    filtered_accel = butter_lowpass_filter(accel_mag, cutoff=4.0, fs=fs)

    # Detect heel-strike peaks
    peaks, _ = find_peaks(filtered_accel, distance=fs * 0.4, prominence=0.30)

    duration_sec = trial_df["time_sec"].iloc[-1] - trial_df["time_sec"].iloc[0]
    cadence = (len(peaks) / duration_sec) * 60.0

    if len(peaks) >= 3:
        stride_times = np.diff(peaks) / fs
        stride_mean = float(np.mean(stride_times))
        stride_std = float(np.std(stride_times))
        stride_cv = (stride_std / stride_mean) * 100.0 if stride_mean > 0 else 0.0
    else:
        stride_mean, stride_cv = 1.0, 0.0

    return {
        "cadence_spm": cadence,
        "stride_mean_s": stride_mean,
        "stride_variability_cv": stride_cv,
        "rms_accel": float(np.sqrt(np.mean(filtered_accel**2))),
        "peak_impact": float(np.max(filtered_accel)),
        "mean_rot_velocity": float(np.mean(gyro_mag)),
        "rot_std": float(np.std(gyro_mag)),
        "subject_id": trial_df["subject_id"].iloc[0],
        "label": trial_df["label"].iloc[0]
    }

# Load generated raw dataset
raw_df = pd.read_csv("imu_gait_raw_dataset.csv")

feature_rows = []
for trial_id, group in raw_df.groupby("trial_id"):
    feat = extract_features_from_trial(group)
    feature_rows.append(feat)

features_df = pd.DataFrame(feature_rows)
print("\nExtracted Biomechanical Feature Summary:")
print(features_df[["cadence_spm", "stride_mean_s", "stride_variability_cv", "label"]].head(10))

# Train / Evaluate Model
X = features_df.drop(columns=["subject_id", "label"])
y = features_df["label"]

clf = RandomForestClassifier(n_estimators=50, random_state=42)
clf.fit(X, y)
preds = clf.predict(X)

print("\nModel Training Results on Raw Dataset:")
print(classification_report(y, preds))
