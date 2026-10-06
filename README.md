# 🏃‍♂️ AI-Driven Biomechanical Gait Analysis & Fall Risk Assessment

[![Python](https://img.shields.io/badge/Python-3.9%20%7C%203.10%20%7C%203.11-blue?logo=python)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Domain](https://img.shields.io/badge/Domain-Biomedical%20Engineering%20%26%20Kinematics-orange)](#)

An intelligent biomechanics and signal processing pipeline developed to analyze wearable inertial measurement unit (IMU) telemetry, isolate heel-strike gait cycles, extract clinical stability metrics, and classify fall risk using machine learning.

---

## 📌 Project Overview & Clinical Context

Dynamic gait stability and stride regularity provide critical diagnostic biomarkers for neuromuscular disorders, post-stroke rehabilitation, and fall prevention in elderly populations. 

This repository provides an end-to-end framework that takes tri-axial accelerometer and gyroscope streams, suppresses mechanical tremor and high-frequency motion artifacts using a zero-phase digital filter, identifies spatiotemporal heel-strike events, and maps these signals to quantitative biomechanical indicators.

---

## 🔬 Biomechanical Indicators & Signal Processing

### 1. Signal Conditioning
Raw tri-axial acceleration ($a_x, a_y, a_z$) and angular velocity ($\omega_{\text{roll}}, \omega_{\text{pitch}}, \omega_{\text{yaw}}$) signals are combined into Euclidean norm vector magnitudes:

$$\Vert{}a(t)\Vert{} = \sqrt{a_x^2 + a_y^2 + a_z^2}$$

To eliminate high-frequency motion noise while preserving physiological gait dynamics without phase distortion, signals pass through a **4th-order zero-phase forward-backward Butterworth low-pass filter** ($f_c = 4.0\text{ Hz}$, $f_s = 50\text{ Hz}$).

### 2. Extracted Features
* **Temporal Kinematics:** 
  * **Cadence ($SPM$):** Steps per minute calculated via automated peak detection over temporal heel-strike events.
  * **Mean Stride Duration (s):** Average time interval between consecutive ipsilateral heel strikes.
* **Dynamic Postural Stability:**
  * **Stride Time Variability ($CV\%$):** Coefficient of variation ($\frac{\sigma}{\mu} \times 100$), a key predictor of dynamic balance impairment.
  * **RMS Acceleration:** Root Mean Square of vertical/resultant acceleration reflecting dynamic postural sway.
* **Impact & Rotational Dynamics:**
  * **Peak Impact Acceleration ($g$):** Heel-strike ground impact shock magnitude.
  * **Mean & Variance of Angular Velocity:** Rotational limb stability during the swing phase.

---

## 📁 Repository Structure

```text
ai-gait-kinematics-fall-detector/
├── data/
│   └── imu_gait_raw_dataset.csv       # Pre-loaded 50 Hz tri-axial IMU telemetry
├── src/
│   ├── test_pipeline.py              # Loads CSV, filters, extracts metrics, trains & evaluates RF
│   └── biomechanics_gait_analyzer.py # Standalone feature extractor & synthetic benchmarking
├── requirements.txt                  # Environment dependencies
└── README.md                         # Project documentation & clinical background
