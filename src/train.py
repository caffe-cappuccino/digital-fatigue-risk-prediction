import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# ================= LOAD DATA =================
df = pd.read_csv("data/digital_fatigue.csv")

# ================= CREATE FATIGUE SCORE (0–100) =================
# This represents HOW MUCH fatigue may occur

raw_score = (
    df["screen_time_hours"] * 6 +
    df["continuous_usage_minutes"] * 0.08 +
    df["night_usage_hours"] * 8 -
    df["breaks_per_day"] * 4 -
    df["sleep_hours"] * 7 +
    df["eye_strain_level"] * 10 +
    df["task_switching_rate"] * 1.5
)

# Normalize to 0–100
df["fatigue_score"] = (
    (raw_score - raw_score.min()) /
    (raw_score.max() - raw_score.min())
) * 100

# ================= FEATURES & TARGET =================
X = df[
    [
        "screen_time_hours",
        "continuous_usage_minutes",
        "night_usage_hours",
        "breaks_per_day",
        "sleep_hours",
        "eye_strain_level",
        "task_switching_rate",
    ]
]

y = df["fatigue_score"]

# ================= TRAIN / TEST SPLIT =================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ================= MODEL =================
model = RandomForestRegressor(
    n_estimators=300,
    max_depth=10,
    random_state=42
)

model.fit(X_train, y_train)

# ================= EVALUATION =================
preds = model.predict(X_test)
mae = mean_absolute_error(y_test, preds)

print(f"Model trained successfully")
print(f"Mean Absolute Error (MAE): {mae:.2f}")

# ================= SAVE MODEL =================
joblib.dump(model, "model/fatigue_model.pkl")
print("Saved model as model/fatigue_model.pkl")
