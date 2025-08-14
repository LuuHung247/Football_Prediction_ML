import os
import base64
import json as _json
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
MODEL_HOME_PATH = os.path.join(os.getcwd(), "models", "home_rf_model.pkl")
MODEL_AWAY_PATH = os.path.join(os.getcwd(), "models", "away_rf_model.pkl")
FEATURE_COLS_PATH = os.path.join(os.getcwd(), "models", "feature_cols.json")
TEAM_ENCODER_PATH = os.path.join(os.getcwd(), "models", "team_encoder.json")
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


def _safe_name(name: str) -> str:
    return (
        str(name).lower()
        .replace(" ", "_")
        .replace("/", "-")
        .replace("&", "and")
        .replace(".", "")
        .replace("'", "")
    )


def get_logo_path(team_name: str) -> str:
    logos_dir = os.path.join(os.getcwd(), "assets", "logos")

    # Known aliases for dataset naming vs. file naming
    alias_map = {
        "nott'm_forest": "nottingham_forest",
        "man_utd": "man_united",
        "man_city": "man_city",
    }

    base = _safe_name(team_name)

    # Variant that preserves apostrophes (older downloaded filenames)
    keep_apostrophe = (
        str(team_name).lower().replace(" ", "_").replace("/", "-").replace("&", "and").replace(".", "")
    )

    candidates = [
        base,
        alias_map.get(base, base),
        keep_apostrophe,
    ]

    for stem in candidates:
        fpath = os.path.join(logos_dir, f"{stem}.png")
        if os.path.exists(fpath):
            return fpath

    # Fallback to the first expected path
    return os.path.join(logos_dir, f"{base}.png")


def render_logo(team_name: str, box_size: int = 120):
    path = get_logo_path(team_name)
    if not os.path.exists(path):
        return
    try:
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        html = f'''<div style="width:{box_size}px;height:{box_size}px;display:flex;align-items:center;justify-content:center;">
  <img src="data:image/png;base64,{b64}" style="max-width:100%;max-height:100%;object-fit:contain;"/>
</div>'''
        st.markdown(html, unsafe_allow_html=True)
    except Exception:
        # fallback to streamlit image with fixed width
        st.image(path, caption=team_name, width=box_size)


def _logo_b64(team_name: str) -> str:
    path = get_logo_path(team_name)
    if not os.path.exists(path):
        return ""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def render_scoreboard(home_team: str, away_team: str, g_home: int, g_away: int):
    b64_home = _logo_b64(home_team)
    b64_away = _logo_b64(away_team)
    html = f'''
    <div style="display:flex;justify-content:center;width:100%;">
      <div style="margin-top:8px;border-radius:12px;background:#1f2937;padding:16px;max-width:680px;width:100%;">
        <div style="display:grid;grid-template-columns:1fr auto 1fr;align-items:center;gap:12px;color:#e5e7eb;">
          <div style="text-align:center;">
            <img src="data:image/png;base64,{b64_home}" style="width:64px;height:64px;object-fit:contain;display:block;margin:0 auto 8px;"/>
            <div style="font-size:16px;">{home_team}</div>
          </div>
          <div style="text-align:center;">
            <div style="font-weight:700;font-size:52px;line-height:1;letter-spacing:1px;">
              {g_home}
              <span style="opacity:.7;margin:0 16px;">-</span>
              {g_away}
            </div>
          </div>
          <div style="text-align:center;">
            <img src="data:image/png;base64,{b64_away}" style="width:64px;height:64px;object-fit:contain;display:block;margin:0 auto 8px;"/>
            <div style="font-size:16px;">{away_team}</div>
          </div>
        </div>
      </div>
    </div>
    '''
    st.markdown(html, unsafe_allow_html=True)



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

    # Load feature columns used during training
    if not os.path.exists(FEATURE_COLS_PATH):
        st.error("Thiếu file models/feature_cols.json. Vui lòng dump metadata từ notebook.")
        st.stop()
    try:
        with open(FEATURE_COLS_PATH, "r") as f:
            feature_cols = _json.load(f).get("feature_cols", [])
    except Exception:
        feature_cols = []
    if not feature_cols:
        st.error("Danh sách cột huấn luyện trống. Vui lòng dump lại feature_cols từ notebook.")
        st.stop()

    # Load team encoder mapping {team_name -> code}
    if not os.path.exists(TEAM_ENCODER_PATH):
        st.error("Thiếu file models/team_encoder.json. Vui lòng dump metadata từ notebook.")
        st.stop()
    try:
        with open(TEAM_ENCODER_PATH, "r") as f:
            team_to_code = _json.load(f).get("to_code", {})
    except Exception:
        team_to_code = {}
    if not team_to_code:
        st.warning("Không có mapping team->code. Sẽ cố gắng suy luận, nhưng nên dump đúng từ notebook.")

    return df, model_home, model_away, feature_cols, team_to_code


def _mean_or_nan(series: pd.Series) -> float:
    series = series.dropna()
    if series.empty:
        return np.nan
    return float(series.mean())


def build_input_row(
    df: pd.DataFrame,
    home_team: str,
    away_team: str,
    feature_cols: List[str],
    team_to_code: Dict[str, int],
) -> pd.DataFrame:
    """Construct a single-row feature DataFrame for a given matchup by aggregating
    team-specific historical statistics from the dataset.

    Heuristics:
    - Columns ending with _H: take mean over matches where team played at Home.
    - Columns ending with _A: take mean over matches where team played Away.
    - Other numeric columns: prefer head-to-head mean for this pairing; fallback to
      per-team means (home or away as available); else global mean.
    """
    feature_columns: List[str] = [c for c in df.columns if c not in EXCLUDE_COLUMNS]

    row: Dict[str, float] = {}

    # Ensure categorical fields present
    for cat in CATEGORICAL_FEATURES:
        if cat not in feature_columns:
            feature_columns.append(cat)

    # Populate categorical names
    row["HomeTeam"] = home_team
    row["AwayTeam"] = away_team
    # And codes expected by the model
    row["HomeTeam_code"] = float(team_to_code.get(home_team, np.nan))
    row["AwayTeam_code"] = float(team_to_code.get(away_team, np.nan))

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

    # Build DataFrame and align to training feature columns order
    row_df = pd.DataFrame([row])
    # Ensure all required columns exist
    for col in feature_cols:
        if col not in row_df.columns:
            row_df[col] = 0.0
    # Select and order
    row_df = row_df[feature_cols]
    # Coerce to numeric where possible
    for col in row_df.columns:
        if col not in CATEGORICAL_FEATURES:
            row_df[col] = pd.to_numeric(row_df[col], errors="coerce")
    # Fill any NaNs with column medians (fallback 0.0)
    row_df = row_df.fillna(row_df.median(numeric_only=True)).fillna(0.0)
    return row_df


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
        df, model_home, model_away, feature_cols, team_to_code = load_models_and_metrics(data_path)
        st.session_state["_cache_models"] = (df, model_home, model_away, feature_cols, team_to_code)

df, model_home, model_away, feature_cols, team_to_code = st.session_state.get(
    "_cache_models", load_models_and_metrics(data_path)
)

# Only allow teams present in the Test season 2024-07-01 → 2025-06-30
test_start = pd.Timestamp("2024-07-01")
test_end = pd.Timestamp("2025-06-30")
df_test = df[(df["Date"] >= test_start) & (df["Date"] <= test_end)]
teams_sorted = sorted(
    set(df_test.get("HomeTeam", pd.Series(dtype=str)).dropna().unique())
    | set(df_test.get("AwayTeam", pd.Series(dtype=str)).dropna().unique())
)

main_col = st.columns([1, 2, 1])[1]
with main_col:
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

    # Reset score on team change
    teams_key = f"{home_team}__{away_team}" if (home_team and away_team) else None
    prev_key = st.session_state.get("_teams_key")
    if teams_key and teams_key != prev_key:
        st.session_state["_teams_key"] = teams_key
        st.session_state["_score_home"] = None
        st.session_state["_score_away"] = None

    if predict_btn and home_team and away_team and home_team != away_team:
        X_row = build_input_row(df, home_team, away_team, feature_cols, team_to_code)
        g_home, g_away = predict_score(model_home, model_away, X_row)
        st.session_state["_score_home"] = g_home
        st.session_state["_score_away"] = g_away

    disp_home = st.session_state.get("_score_home")
    disp_away = st.session_state.get("_score_away")
    disp_home = disp_home if disp_home is not None else "?"
    disp_away = disp_away if disp_away is not None else "?"
    if home_team and away_team and home_team != away_team:
        render_scoreboard(home_team, away_team, disp_home, disp_away)

    # Head-to-head (last 5, exclude Test period)
    if home_team and away_team and home_team != away_team:
        try:
            mask_pair = (
                ((df.get("HomeTeam") == home_team) & (df.get("AwayTeam") == away_team))
                |
                ((df.get("HomeTeam") == away_team) & (df.get("AwayTeam") == home_team))
            )
            mask_time = df.get("Date") < test_start
            df_h2h = df[mask_pair & mask_time][["Date", "HomeTeam", "AwayTeam", TARGET_HOME, TARGET_AWAY]].dropna()
            df_h2h = df_h2h.sort_values("Date", ascending=False).head(5).copy()
            df_h2h["Date"] = pd.to_datetime(df_h2h["Date"], errors="coerce").dt.date.astype(str)
            df_h2h = df_h2h.rename(columns={TARGET_HOME: "FT_Home", TARGET_AWAY: "FT_Away"})
            st.subheader("Đối đầu gần nhất (tối đa 5 trận)")
            st.dataframe(df_h2h.reset_index(drop=True), use_container_width=True)
        except Exception:
            st.info("Không lấy được lịch sử đối đầu.")

    st.markdown("---")
    st.subheader("Đầu vào ước lượng (từ thống kê lịch sử)")
    if predict_btn and home_team and away_team and home_team != away_team:
        st.dataframe(
            build_input_row(df, home_team, away_team, feature_cols, team_to_code).T.rename(columns={0: "Giá trị"}),
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


