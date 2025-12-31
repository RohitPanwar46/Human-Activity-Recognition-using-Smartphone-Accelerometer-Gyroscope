# 🧠 Human Activity Recognition using Smartphone Sensors

> A session-separated Human Activity Recognition pipeline using smartphone accelerometer and gyroscope data.

A complete **Human Activity Recognition (HAR)** system built using smartphone accelerometer and gyroscope data, organized as a realistic, session-separated ML project with clean preprocessing, feature extraction, and evaluation.

---

## Why this version is **much better**

✔ Matches your **real folder structure**  
✔ Shows **train vs test separation** (huge plus)  
✔ Clearly separates:
- pipeline code (`src/har`)
- trained models
✔ Looks like a **serious ML project**, not a tutorial

---

## 📌 Project Overview

This repository classifies human activities using time-series sensor data collected from a smartphone placed in a front pocket. The project prioritizes correct sensor handling, honest session-separated evaluation, and reproducible experiments.

Recognized activities:
- Sitting
- Standing
- Walking
- Running
- Stairs (up/down)

---

## 🛠️ Sensors Used

- **Accelerometer** (with gravity)
- **Gyroscope**

Both sensors are synchronized and merged using timestamps before feature extraction.

---

## 📂 Dataset Collection & Preprocessing

### Data Collection
- Data recorded using a smartphone (Android)
- Fixed phone placement (front pocket)
- Separate recording sessions for **training** and **testing** (session-separated)

### Preprocessing Steps
1. Merge accelerometer and gyroscope streams using timestamp alignment
2. Remove sensor warm-up noise
3. Trim initial and final seconds to remove non-representative motion
4. Remove rows with missing values or invalid zero readings
5. Use only cleaned, merged sensor data for modeling

This ensures **no data leakage** and realistic generalization.

---

## ⏱️ Windowing Strategy

- **Window size**: 2 seconds
- **Sampling rate**: ~50 Hz
- **Samples per window**: ~100
- **Overlap**: 50%

Each window is an input example for feature extraction.

---

## 📊 Feature Engineering

For each window, statistical time-domain features are extracted.

### Accelerometer Features
- Mean (x, y, z)
- Standard deviation (x, y, z)
- Mean acceleration magnitude
- Std acceleration magnitude

### Gyroscope Features
- Mean (x, y, z)
- Standard deviation (x, y, z)
- Mean gyroscope magnitude
- Std gyroscope magnitude

**Total features per window:** ~16 (depends on exact implementation)

---

## 🤖 Model Selection

### Final Model: Logistic Regression

- Strong performance due to well-engineered features
- Interpretable and stable
- Simpler model reduces overfitting risk

Random Forest and other models were explored during experiments.

---

## 📈 Evaluation Strategy

- **Session-level train/test split** (different recording sessions)
- No random shuffling that mixes sessions
- No leakage from overlapping windows across train/test

This evaluation mirrors realistic deployment scenarios.

---

## ✅ Results (example)

**Overall Accuracy:** ~95% (depends on experiment and split)

Observed behaviors:
- Sitting & standing: high accuracy
- Running: distinct gyroscope signatures
- Walking vs stairs: expected confusion in some cases

---

## 📁 Project Structure (reflects workspace)

A compact view of the repository (mirrors workspace layout):

```text
.
├── data
│   ├── running train 01
│   │   ├── Accelerometer.csv
│   │   ├── Gyroscope.csv
│   │   └── trimmed_merged_data.csv
│   ├── running test 01
│   │   ├── Accelerometer.csv
│   │   ├── Gyroscope.csv
│   │   └── trimmed_merged_data.csv
│   ├── walking train 01
│   ├── walking test 01
│   ├── standing train 01
│   ├── standing test 01
│   ├── stairs train 01
│   ├── stairs test 01
│   ├── sitting 01 train
│   └── sitting test 01
├── src
│   └── har
│       ├── data_preprocessing.py
│       ├── feature_extaction.py
│       ├── main.py
│       └── training_models.py
└── models
        ├── LogisticRegressionModel.pkl
```

---

## 🚀 Future Improvements

- Add frequency-domain features (FFT)
- Build orientation-invariant features
- Add real-time inference demo
- Expand multi-user and placement variability testing

---

## 🧠 Key Takeaway

Strong feature engineering and session-separated evaluation produce reliable, deployable HAR models that generalize better than flashy benchmarks.

---

## ⚡ Quick start — preprocess & train

1. Install minimal dependencies (use your environment manager of choice):

```bash
pip install pandas scikit-learn joblib
```

2. Preprocess raw sensor CSVs (creates `trimmed_merged_data.csv` in each data folder):

```bash
python -u "src/har/data_preprocessing.py"
```

3. Train the model and save it to `models/LogisticRegression.pkl`:

```bash
python -u "src/har/training_models.py"
```

---

## Using `data_preprocessing.py` (API)

The script also exposes simple functions you can import and call from Python:

```python
from har.data_preprocessing import merge_accelerometer_gyroscope, trim_df, save_df_to_csv

# Merge individual folder files (pass the folder path ending with /)
df = merge_accelerometer_gyroscope("data/walking train 01/")

# Trim by seconds from start/end
trimmed = trim_df(df, start_trim=5, end_trim=5)

# Or run the helper that scans `data/` and writes `trimmed_merged_data.csv` for each session
all_dfs = save_df_to_csv()
```

Notes:
- `merge_accelerometer_gyroscope` expects the two files `Accelerometer.csv` and `Gyroscope.csv` inside the folder you pass.
- `save_df_to_csv()` looks for folders under `data/` that contain both files and writes `trimmed_merged_data.csv` into each folder.


