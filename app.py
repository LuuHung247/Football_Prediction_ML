import os
import json
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from joblib import load


# -------------------------
# Config & Constants
# -------------------------
st.set_page_config(
    page_title="Dự đoán tỉ số bóng đá (XGBoost)",
    page_icon="⚽",
    layout="wide",
)

DEFAULT_DATA_PATH = os.path.join(os.getcwd(), "dataset", "data.csv")
MODEL_HOME_PATH = os.path.join(os.getcwd(), "models", "home_model.pkl")
MODEL_AWAY_PATH = os.path.join(os.getcwd(), "models", "away_model.pkl")
TARGET_HOME = "Full Time Home Goals"
TARGET_AWAY = "Full Time Away Goals"
CATEGORICAL_FEATURES = ["HomeTeam", "AwayTeam"]
EXCLUDE_COLUMNS = {"Date", TARGET_HOME, TARGET_AWAY}


# -------------------------
# Utils
# -------------------------
def read_dataset(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Normalize column names (strip spaces)
    df.columns = [c.strip() for c in df.columns]

    # Parse Date if present
    if "Date" in df.columns:
        # Keep original for reference but we won't use it as a feature
        # Attempt day-first (as sample shows dd-mm-yy)
        try:
            df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
        except Exception:
            pass

    # Coerce numerics for all non-categorical, non-date columns
    for col in df.columns:
        if col in CATEGORICAL_FEATURES or col == "Date":
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def split_features_targets(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    # Keep for building input rows and any light evaluation if needed
    df = df.dropna(subset=[TARGET_HOME, TARGET_AWAY])
    feature_columns: List[str] = [c for c in df.columns if c not in EXCLUDE_COLUMNS]
    X = df[feature_columns].copy()
    y_home = df[TARGET_HOME].astype(float)
    y_away = df[TARGET_AWAY].astype(float)
    return X, y_home, y_away




@st.cache_resource(show_spinner=True)
def load_models_and_metrics(csv_path: str):
    df = read_dataset(csv_path)
    if "Date" not in df.columns:
        st.error("Cột 'Date' không tồn tại trong dữ liệu.")
        st.stop()
    df = df.dropna(subset=["Date"]).copy()

    if not (os.path.exists(MODEL_HOME_PATH) and os.path.exists(MODEL_AWAY_PATH)):
        st.error("Chưa có mô hình. Vui lòng mở notebook 'eda_and_train.ipynb' để huấn luyện và lưu mô hình vào ./models/.")
        st.stop()
    model_home = load(MODEL_HOME_PATH)
    model_away = load(MODEL_AWAY_PATH)

    metrics_path = os.path.join(os.getcwd(), "models", "metrics.json")
    metrics: Dict[str, Any]
    if os.path.exists(metrics_path):
        try:
            import json as _json
            with open(metrics_path, "r") as f:
                metrics = _json.load(f)
        except Exception:
            metrics = {}
    else:
        metrics = {}

    return df, model_home, model_away, metrics


def _mean_or_nan(series: pd.Series) -> float:
    series = series.dropna()
    if series.empty:
        return np.nan
    return float(series.mean())


def build_input_row(df: pd.DataFrame, home_team: str, away_team: str) -> pd.DataFrame:
    """Construct a single-row feature DataFrame for a given matchup by aggregating
    team-specific historical statistics from the dataset.

    Heuristics:
    - Columns ending with _H: take mean over matches where team played at Home.
    - Columns ending with _A: take mean over matches where team played Away.
    - Other numeric columns: prefer head-to-head mean for this pairing; fallback to
      per-team means (home or away as available); else global mean.
    """
    feature_columns: List[str] = [
        c for c in df.columns if c not in EXCLUDE_COLUMNS
    ]

    row: Dict[str, float] = {}

    # Ensure categorical fields present
    for cat in CATEGORICAL_FEATURES:
        if cat not in feature_columns:
            feature_columns.append(cat)

    # Populate categorical
    row["HomeTeam"] = home_team
    row["AwayTeam"] = away_team

    # Numeric columns
    numeric_cols = [
        c for c in feature_columns if c not in CATEGORICAL_FEATURES and c != "Date"
    ]

    # Pre-computed masks for speed
    mask_home = df["HomeTeam"] == home_team
    mask_away = df["AwayTeam"] == away_team
    mask_h2h = mask_home & mask_away

    for col in numeric_cols:
        if col.endswith("_H"):
            val = _mean_or_nan(df.loc[mask_home, col])
        elif col.endswith("_A"):
            val = _mean_or_nan(df.loc[mask_away, col])
        else:
            # Prefer direct head-to-head context for this pairing
            val = _mean_or_nan(df.loc[mask_h2h, col])
            if np.isnan(val):
                # Fallback to team-position means
                v_home = _mean_or_nan(df.loc[mask_home, col])
                v_away = _mean_or_nan(df.loc[mask_away, col])
                if np.isnan(v_home) and np.isnan(v_away):
                    val = _mean_or_nan(df[col])
                else:
                    val = float(np.nanmean([v_home, v_away]))

        if np.isnan(val):
            # Ultimate fallback: global median
            try:
                val = float(df[col].median())
            except Exception:
                val = 0.0
        row[col] = val

    return pd.DataFrame([row])[feature_columns]


def predict_score(model_home: Any, model_away: Any, X_row: pd.DataFrame) -> Tuple[int, int]:
    pred_home = float(model_home.predict(X_row)[0])
    pred_away = float(model_away.predict(X_row)[0])
    # Clamp and round
    pred_home = max(0.0, min(10.0, pred_home))
    pred_away = max(0.0, min(10.0, pred_away))
    return int(round(pred_home)), int(round(pred_away))


# -------------------------
# UI
# -------------------------
st.title("⚽ Dự đoán tỉ số bóng đá bằng XGBoost")
st.caption(
    "Huấn luyện mô hình từ dữ liệu lịch sử và dự đoán tỉ số cho cặp đấu được chọn."
)

data_path = DEFAULT_DATA_PATH

if "_cache_models" not in st.session_state:
    with st.spinner("Đang tải mô hình đã huấn luyện..."):
        df, model_home, model_away, metrics = load_models_and_metrics(data_path)
        st.session_state["_cache_models"] = (df, model_home, model_away, metrics)

df, model_home, model_away, metrics = st.session_state.get("_cache_models", load_models_and_metrics(data_path))

# Only allow teams present in the Test season 2024-07-01 → 2025-06-30
test_start = pd.Timestamp("2024-07-01")
test_end = pd.Timestamp("2025-06-30")
df_test = df[(df["Date"] >= test_start) & (df["Date"] <= test_end)]
teams_sorted = sorted(
    set(df_test.get("HomeTeam", pd.Series(dtype=str)).dropna().unique())
    | set(df_test.get("AwayTeam", pd.Series(dtype=str)).dropna().unique())
)

col_left, col_right = st.columns([1, 1])
with col_left:
    st.subheader("Chọn cặp đấu (chỉ đội trong tập Test 2024-2025)")
    if len(teams_sorted) == 0:
        st.error("Không tìm thấy đội nào trong tập Test 2024-2025.")
        home_team, away_team, predict_btn = None, None, False
    else:
        home_team = st.selectbox(
            "Đội nhà",
            options=teams_sorted,
            index=0,
            key="home_select",
        )
        away_options = [t for t in teams_sorted if t != home_team]
        if not away_options:
            st.warning("Tập test chỉ có 1 đội, không thể chọn đội khách khác.")
            away_team = None
        else:
            away_team = st.selectbox(
                "Đội khách",
                options=away_options,
                index=0,
                key="away_select",
            )
        predict_btn = st.button("Dự đoán tỉ số")

with col_right:
    st.subheader("Chất lượng mô hình")
    st.caption("Theo mùa: Val 2023-2024, Test 2024-2025 (từ notebook)")

    def _fmt(mdict: Dict, split: str, side: str, key: str) -> str:
        try:
            val = mdict[split][side][key]
            if val is None:
                return "N/A"
            return f"{float(val):.3f}"
        except Exception:
            return "N/A"

    st.markdown("Val 2023-2024")
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("MAE (Home)", _fmt(metrics, "val", "home", "MAE"))
        st.metric("MAE (Away)", _fmt(metrics, "val", "away", "MAE"))
    with m2:
        st.metric("RMSE (Home)", _fmt(metrics, "val", "home", "RMSE"))
        st.metric("RMSE (Away)", _fmt(metrics, "val", "away", "RMSE"))
    with m3:
        st.metric("R2 (Home)", _fmt(metrics, "val", "home", "R2"))
        st.metric("R2 (Away)", _fmt(metrics, "val", "away", "R2"))

    st.markdown("Test 2024-2025")
    t1, t2, t3 = st.columns(3)
    with t1:
        st.metric("MAE (Home)", _fmt(metrics, "test", "home", "MAE"))
        st.metric("MAE (Away)", _fmt(metrics, "test", "away", "MAE"))
    with t2:
        st.metric("RMSE (Home)", _fmt(metrics, "test", "home", "RMSE"))
        st.metric("RMSE (Away)", _fmt(metrics, "test", "away", "RMSE"))
    with t3:
        st.metric("R2 (Home)", _fmt(metrics, "test", "home", "R2"))
        st.metric("R2 (Away)", _fmt(metrics, "test", "away", "R2"))

st.markdown("---")
output_col1, output_col2 = st.columns([1, 2])

with output_col1:
    st.subheader("Kết quả dự đoán")
    if predict_btn:
        if not home_team or not away_team:
            st.error("Vui lòng chọn đội nhà và đội khách hợp lệ.")
        elif home_team == away_team:
            st.error("Đội nhà và đội khách không được trùng nhau.")
        else:
            X_row = build_input_row(df, home_team, away_team)
            g_home, g_away = predict_score(model_home, model_away, X_row)
            st.success(f"Dự đoán: {home_team} {g_home} - {g_away} {away_team}")

with output_col2:
    st.subheader("Đầu vào ước lượng (từ thống kê lịch sử)")
    if predict_btn and home_team and away_team and home_team != away_team:
        st.dataframe(
            build_input_row(df, home_team, away_team).T.rename(columns={0: "Giá trị"}),
            use_container_width=True,
        )

with st.expander("Ghi chú & Hướng dẫn"):
    st.markdown(
        """
        - Ứng dụng sử dụng 2 mô hình XGBoost Regression để dự đoán số bàn của đội nhà và đội khách.
        - Đặc trưng đầu vào gồm: one-hot của `HomeTeam`/`AwayTeam` và các cột số trong dữ liệu (ví dụ Elo, chuỗi trận, trung bình bàn thắng/thua,...).
        - Khi dự đoán cho cặp đấu mới, các đặc trưng số được ước lượng bằng trung bình có trọng số theo vai trò (Home/ Away) và đối đầu trực tiếp nếu có.
        - Các kết quả chỉ mang tính tham khảo, không dùng cho mục đích cá cược.
        """
    )


