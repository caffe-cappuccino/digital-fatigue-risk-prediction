"""
Robust training script for Digital Fatigue Model
- Safe model saving (atomic write)
- Strong validation
- Guaranteed sklearn model with .predict()
- Designed to avoid corrupted pickle files
"""

import os
import sys
import tempfile
import joblib
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# ================= CONFIG =================
DATA_PATH = "data/digital_fatigue.csv"
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "fatigue_model.pkl")

RANDOM_STATE = 42

# ================= SAFETY CHECKS =================
print("🔍 Starting training pipeline...")

if not os.path.exists(DATA_PATH):
    sys.exit(f"❌ Dataset not found at {DATA_PATH}")

os.makedirs(MODEL_DIR, exist_ok=True)

# ================= LOAD DATA =================
df = pd.read_csv(DATA_PATH)

required_cols = [
    "screen_time_hours",
    "continuous_usage_minutes",
    "night_usage_hours",
    "breaks_per_day",
    "sleep_hours",
    "eye_strain_level",
    "task_switching_rate",
]

for col in required_cols:
    if col not in df.columns:
        sys.exit(f"❌ Missing required column: {col}")

df = df.dropna().reset_index(drop=True)

print(f"✅ Dataset loaded: {df.shape[0]} rows")

# ================= TARGET ENGINEERING =================
# Continuous fatigue score (0–100)

raw_score = (
    df["screen_time_hours"] * 6
    + df["continuous_usage_minutes"] * 0.08
    + df["night_usage_hours"] * 8
    - df["breaks_per_day"] * 4
    - df["sleep_hours"] * 7
    + df["eye_strain_level"] * 10
    + df["task_switching_rate"] * 1.5
)

raw_min, raw_max = raw_score.min(), raw_score.max()

df["fatigue_score"] = ((raw_score - raw_min) / (raw_max - raw_min)) * 100

# ================= FEATURES / LABEL =================
X = df[required_cols]
y = df["fatigue_score"]

# ================= SPLIT =================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)

# ================= MODEL =================
model = RandomForestRegressor(
    n_estimators=300,
    max_depth=12,
    min_samples_leaf=3,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

print("🧠 Training model...")
model.fit(X_train, y_train)

# ================= VALIDATION =================
preds = model.predict(X_test)

mae = mean_absolute_error(y_test, preds)
print(f"📊 Validation MAE: {mae:.2f}")

# Sanity prediction check
test_pred = model.predict(X_train.iloc[:1])
assert isinstance(test_pred, (np.ndarray, list)), "Prediction output invalid"

print("✅ Model prediction sanity check passed")

# ================= SAFE SAVE (ATOMIC WRITE) =================
# Prevents corrupted .pkl files

with tempfile.NamedTemporaryFile(delete=False) as tmp:
    joblib.dump(model, tmp.name)
    temp_path = tmp.name

os.replace(temp_path, MODEL_PATH)

# ================= FINAL VERIFICATION =================
loaded_model = joblib.load(MODEL_PATH)

assert hasattr(loaded_model, "predict"), "Saved model missing .predict()"

print(f"💾 Model safely saved to {MODEL_PATH}")
print("🎉 Training completed successfully")
