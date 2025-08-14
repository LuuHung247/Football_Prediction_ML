import os
import base64
import json as _json
from typing import Any, Dict, List, Tuple
from pathlib import Path
from collections import defaultdict, deque

import numpy as np
import pandas as pd
import streamlit as st
from joblib import load
from sklearn.preprocessing import StandardScaler, LabelEncoder


# -------------------------
# Config & Constants
# -------------------------
st.set_page_config(
    page_title="Dự đoán tỉ số bóng đá",
    page_icon="⚽",
    layout="wide",
)

DEFAULT_DATA_PATH = os.path.join(os.getcwd(), "dataset", "data_baseline.csv")
H2H_DATA_PATH = os.path.join(os.getcwd(), "dataset", "data.csv")
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

# -------------------------
# Create Custome dataset
# -------------------------

def process_full_features(df_base, new_df=None, k=5, init_elo=1500, adv_const=0.05, elo_scale=150.0, K=20):
    df_base["Date"] = pd.to_datetime(df_base["Date"], dayfirst=True)
    
    if new_df is not None:
        new_df["Date"] = pd.to_datetime(new_df["Date"], dayfirst=True)
        df_base = pd.concat([df_base, new_df], ignore_index=True)
    
    df = df_base.sort_values("Date").reset_index(drop=True)
    df.index.name = "MatchID"

    ### Part 1: Win rate home/away
    long = pd.concat([
        pd.DataFrame({
            "MatchID": df.index,
            "Date": df["Date"],
            "Team": df["HomeTeam"],
            "is_home": True,
            "GF": df["Full Time Home Goals"],
            "GA": df["Full Time Away Goals"]
        }),
        pd.DataFrame({
            "MatchID": df.index,
            "Date": df["Date"],
            "Team": df["AwayTeam"],
            "is_home": False,
            "GF": df["Full Time Away Goals"],
            "GA": df["Full Time Home Goals"]
        })
    ], ignore_index=True)

    long["ResultNum"] = np.sign(long["GF"] - long["GA"])
    long["WinFlag"] = (long["ResultNum"] == 1).astype(float)
    long = long.sort_values(["Team", "Date", "MatchID"]).reset_index(drop=True)

    # Win rate k trận sân nhà
    def compute_win_rate_k(df, is_home=True):
        result = []
        for team, g in df[df["is_home"] == is_home].groupby("Team", sort=False):
            wins_hist = []
            for _, r in g.iterrows():
                rate = sum(wins_hist) / len(wins_hist) if wins_hist else 0.0
                result.append((r["MatchID"], team, rate))
                wins_hist.append(r["WinFlag"])
                if len(wins_hist) > k:
                    wins_hist.pop(0)
        return pd.DataFrame(result, columns=["MatchID", "Team", f'{"home" if is_home else "away"}_win_rate_k'])

    home_rate_df = compute_win_rate_k(long, True)
    away_rate_df = compute_win_rate_k(long, False)

    df = df.merge(home_rate_df.rename(columns={"Team": "HomeTeam"}), on=["MatchID", "HomeTeam"], how="left") \
           .merge(away_rate_df.rename(columns={"Team": "AwayTeam"}), on=["MatchID", "AwayTeam"], how="left")
    
    df[["home_win_rate_k", "away_win_rate_k"]] = df[["home_win_rate_k", "away_win_rate_k"]].fillna(0.0)
    df["Home_adv"] = df["home_win_rate_k"] - df["away_win_rate_k"] + adv_const
    df["Home_adv_elo"] = elo_scale * df["Home_adv"]

    ### Part 2: ELO
    teams = pd.unique(pd.concat([df["HomeTeam"], df["AwayTeam"]], ignore_index=True))
    elo = {team: float(init_elo) for team in teams}

    eH, eA = [], []
    for _, row in df.iterrows():
        h, a = row["HomeTeam"], row["AwayTeam"]
        Rh, Ra = elo[h], elo[a]
        eH.append(Rh)
        eA.append(Ra)

        h_adv = row["Home_adv_elo"]
        Rdiff = (Rh + h_adv) - Ra
        We = 1.0 / (1.0 + 10 ** (-Rdiff / 400.0))
        hg, ag = row["Full Time Home Goals"], row["Full Time Away Goals"]
        W = 1.0 if hg > ag else (0.5 if hg == ag else 0.0)

        margin = abs(hg - ag)
        G = (np.log1p(margin) if margin > 0 else 1.0) * (2.2 / (0.001 * abs(Rdiff) + 2.2))
        delta = K * G * (W - We)

        elo[h] += delta
        elo[a] -= delta

    df["Elo_H_before"] = eH
    df["Elo_A_before"] = eA

    ### Part 3: Team split stats (WinStreak, Goals scored/conceded, etc.)
    def expand_team_view(df):
        home = pd.DataFrame({
            "MatchID": df.index, "Date": df["Date"], "Team": df["HomeTeam"],
            "Opponent": df["AwayTeam"], "is_home": True,
            "GS": df["Full Time Home Goals"], "GA": df["Full Time Away Goals"]
        })
        away = pd.DataFrame({
            "MatchID": df.index, "Date": df["Date"], "Team": df["AwayTeam"],
            "Opponent": df["HomeTeam"], "is_home": False,
            "GS": df["Full Time Away Goals"], "GA": df["Full Time Home Goals"]
        })
        return pd.concat([home, away], ignore_index=True)

    split = expand_team_view(df)
    split["ResultNum"] = np.sign(split["GS"] - split["GA"])
    split = split.sort_values(["Team", "Date", "MatchID"]).reset_index(drop=True)

    GS_shift = split.groupby("Team")["GS"].shift(1)
    GA_shift = split.groupby("Team")["GA"].shift(1)

    rolling_count = GS_shift.groupby(split["Team"]).rolling(window=k, min_periods=1).count().reset_index(level=0, drop=True)
    split["GS_k"] = GS_shift.groupby(split["Team"]).rolling(window=k, min_periods=1).sum().reset_index(level=0, drop=True)
    split["GA_k"] = GA_shift.groupby(split["Team"]).rolling(window=k, min_periods=1).sum().reset_index(level=0, drop=True)
    split["GD_k"] = split["GS_k"] - split["GA_k"]

    WinFlag = (split["ResultNum"] == 1).astype(int).groupby(split["Team"]).shift(1)
    LossFlag = (split["ResultNum"] == -1).astype(int).groupby(split["Team"]).shift(1)
    split["Wins_k"] = WinFlag.groupby(split["Team"]).rolling(window=k, min_periods=1).sum().reset_index(level=0, drop=True)
    split["Losses_k"] = LossFlag.groupby(split["Team"]).rolling(window=k, min_periods=1).sum().reset_index(level=0, drop=True)
    split["WinRate_k"] = split["Wins_k"] / rolling_count
    split["GS_avg_k"] = split["GS_k"] / rolling_count
    split["GA_avg_k"] = split["GA_k"] / rolling_count

    # Win/Lose Streak
    prev_result = split.groupby("Team")["ResultNum"].shift(1).fillna(0).astype(int)
    win_streak = np.zeros(len(split))
    lose_streak = np.zeros(len(split))
    for _, idxs in split.groupby("Team").indices.items():
        w = l = 0
        for pos in idxs:
            x = int(prev_result.iloc[pos])
            w = w + 1 if x == 1 else 0
            l = l + 1 if x == -1 else 0
            win_streak[pos] = min(w, k)
            lose_streak[pos] = min(l, k)

    split["WinStreak"] = win_streak.astype(int)
    split["LoseStreak"] = lose_streak.astype(int)

    # Merge lại về từng trận đấu
    features = ["MatchID","Team","is_home","GS_k","GA_k","GD_k","Wins_k","Losses_k",
                "WinRate_k","WinStreak","LoseStreak","GS_avg_k","GA_avg_k"]

    home_df = split[split["is_home"]][features].rename(columns={
        "Team":"HomeTeam","GS_k":"GoalsScore_H","GA_k":"GoalsAgainst_H","GD_k":"GoalDifference_H",
        "Wins_k":"Wins_H","Losses_k":"Losses_H","WinRate_k":"WinRate_H","WinStreak":"WinStreak_H",
        "LoseStreak":"LoseStreak_H","GS_avg_k":"GoalsScore_H_avg","GA_avg_k":"GoalsAgainst_H_avg"
    }).drop(columns="is_home")

    away_df = split[~split["is_home"]][features].rename(columns={
        "Team":"AwayTeam","GS_k":"GoalsScore_A","GA_k":"GoalsAgainst_A","GD_k":"GoalDifference_A",
        "Wins_k":"Wins_A","Losses_k":"Losses_A","WinRate_k":"WinRate_A","WinStreak":"WinStreak_A",
        "LoseStreak":"LoseStreak_A","GS_avg_k":"GoalsScore_A_avg","GA_avg_k":"GoalsAgainst_A_avg"
    }).drop(columns="is_home")

    df = (
        df.reset_index()
        .merge(home_df, on=["MatchID","HomeTeam"], how="left")
        .merge(away_df, on=["MatchID","AwayTeam"], how="left")
    )

    ### Part 4: Head2Head
    df_sorted = df.sort_values(["Date","MatchID"]).reset_index()
    matches_hist = defaultdict(deque)
    prev_matchs, h2h_gs_H_total, h2h_gs_A_total, h2h_gs_H_avg, h2h_gs_A_avg = {}, {}, {}, {}, {}

    for _, row in df_sorted.iterrows():
        h, a = row["HomeTeam"], row["AwayTeam"]
        key = tuple(sorted((h, a)))
        past = list(matches_hist[key])

        total = sum(1 if t[0] == h and t[1] == 1 else -1 if t[0] == a and t[1] == 1 else 0 for t in past)
        goals_H_total = sum(t[2] if t[0] == h else t[3] for t in past)
        goals_A_total = sum(t[3] if t[0] == h else t[2] for t in past)
        count = len(past)

        idx = row["index"]
        prev_matchs[idx] = total
        h2h_gs_H_total[idx] = goals_H_total
        h2h_gs_A_total[idx] = goals_A_total
        h2h_gs_H_avg[idx] = goals_H_total / count if count else 0.0
        h2h_gs_A_avg[idx] = goals_A_total / count if count else 0.0

        matches_hist[key].append((h, np.sign(row["Full Time Home Goals"] - row["Full Time Away Goals"]),
                                  row["Full Time Home Goals"], row["Full Time Away Goals"]))

    df["H2H_score"] = df.index.to_series().map(prev_matchs)
    df["H2H_GS_H_total"] = df.index.to_series().map(h2h_gs_H_total)
    df["H2H_GS_A_total"] = df.index.to_series().map(h2h_gs_A_total)
    df["H2H_GS_H_avg"] = df.index.to_series().map(h2h_gs_H_avg)
    df["H2H_GS_A_avg"] = df.index.to_series().map(h2h_gs_A_avg)

    # Final selected columns
    final_cols = [
        "Date","HomeTeam","AwayTeam","Elo_H_before","Elo_A_before", 
        "GoalsScore_H","GoalsAgainst_H","GoalDifference_H",
        "WinStreak_H","LoseStreak_H","Wins_H","Losses_H","WinRate_H",
        "GoalsScore_H_avg", "GoalsAgainst_H_avg", 
        "Home_adv_elo",
        "GoalsScore_A","GoalsAgainst_A","GoalDifference_A",
        "WinStreak_A","LoseStreak_A","Wins_A","Losses_A","WinRate_A",
        "GoalsScore_A_avg", "GoalsAgainst_A_avg",
        "H2H_score","H2H_GS_H_total", "H2H_GS_A_total", "H2H_GS_H_avg",
        "H2H_GS_A_avg", "Full Time Home Goals", "Full Time Away Goals"
    ]
    return df[final_cols].sort_values("Date").reset_index(drop=True)


def prepare_features_for_prediction(
    folder="data_season",
    new_data=None,
    baseline_path="data_extra/new_season.csv",
    k=5,
    init_elo=1500,
    adv_const=0.05,
    elo_scale=150.0,
    K=20
):
    # Load historical data
    files = sorted(Path(folder).glob("*.csv"), key=lambda p: p.stem)
    df_hist_raw = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
    df_hist_raw["Date"] = pd.to_datetime(df_hist_raw["Date"], dayfirst=True)

    # Nếu có new_data
    if new_data is not None:
        new_df = pd.DataFrame(new_data)

        # Nếu không có Date, tự động sinh từ ngày cuối cùng
        if "Date" not in new_df.columns or new_df["Date"].isna().all():
            last_date = df_hist_raw["Date"].max()
            new_df["Date"] = [last_date + pd.Timedelta(days=i+1) for i in range(len(new_df))]
        else:
            new_df["Date"] = pd.to_datetime(new_df["Date"], dayfirst=True)
    else:
        new_df = None

    # Process full historical + new matches
    df_hist = process_full_features(
        df_hist_raw, 
        new_df=new_df, 
        k=k, init_elo=init_elo,
        adv_const=adv_const, elo_scale=elo_scale, K=K
    ).fillna(0)

    # Load baseline
    df_baseline = pd.read_csv(baseline_path)

    # ===== Các bước giữ nguyên như cũ =====
    # Elo
    elo_latest = {}
    for team in pd.unique(pd.concat([df_hist["HomeTeam"], df_hist["AwayTeam"]], ignore_index=True)):
        last_home = df_hist[df_hist["HomeTeam"] == team][["Elo_H_before"]].tail(1)
        last_away = df_hist[df_hist["AwayTeam"] == team][["Elo_A_before"]].tail(1)
        last = pd.concat([last_home.rename(columns={"Elo_H_before": "Elo"}),
                          last_away.rename(columns={"Elo_A_before": "Elo"})])
        elo_latest[team] = last["Elo"].iloc[-1] if not last.empty else init_elo

    df_baseline["Elo_H_before"] = df_baseline["HomeTeam"].map(elo_latest).fillna(init_elo)
    df_baseline["Elo_A_before"] = df_baseline["AwayTeam"].map(elo_latest).fillna(init_elo)

    # Win rate
    def get_win_rate(df, team, is_home=True):
        cond = (df["HomeTeam"] == team) if is_home else (df["AwayTeam"] == team)
        win = (df[cond]["Full Time Home Goals"] > df[cond]["Full Time Away Goals"]) if is_home else \
              (df[cond]["Full Time Away Goals"] > df[cond]["Full Time Home Goals"])
        return win.tail(k).mean() if not win.empty else 0.0

    home_win_rate_k = df_baseline["HomeTeam"].apply(lambda t: get_win_rate(df_hist, t, True))
    away_win_rate_k = df_baseline["AwayTeam"].apply(lambda t: get_win_rate(df_hist, t, False))
    df_baseline["Home_adv_elo"] = (home_win_rate_k - away_win_rate_k + adv_const) * elo_scale

    # Head-to-Head
    def get_h2h_stats(df, h, a):
        matches = df[((df["HomeTeam"] == h) & (df["AwayTeam"] == a)) |
                     ((df["HomeTeam"] == a) & (df["AwayTeam"] == h))].tail(k)
        score = gs_H = gs_A = 0
        for _, r in matches.iterrows():
            is_home = r["HomeTeam"] == h
            result = np.sign(r["Full Time Home Goals"] - r["Full Time Away Goals"])
            result = result if is_home else -result
            score += result
            gs_H += r["Full Time Home Goals"] if is_home else r["Full Time Away Goals"]
            gs_A += r["Full Time Away Goals"] if is_home else r["Full Time Home Goals"]
        n = len(matches)
        return pd.Series({
            "H2H_score": score,
            "H2H_GS_H_total": gs_H,
            "H2H_GS_A_total": gs_A,
            "H2H_GS_H_avg": gs_H / n if n else 0.0,
            "H2H_GS_A_avg": gs_A / n if n else 0.0
        })

    h2h_df = df_baseline.apply(lambda r: get_h2h_stats(df_hist, r["HomeTeam"], r["AwayTeam"]), axis=1)
    df_baseline = pd.concat([df_baseline, h2h_df], axis=1)

    # Merge last stats
    latest_home = df_hist.drop_duplicates("HomeTeam", keep="last").set_index("HomeTeam")
    latest_away = df_hist.drop_duplicates("AwayTeam", keep="last").set_index("AwayTeam")

    home_features = [
        "GoalsScore_H", "GoalsAgainst_H", "GoalDifference_H",
        "WinStreak_H", "LoseStreak_H", "Wins_H", "Losses_H", "WinRate_H",
        "GoalsScore_H_avg", "GoalsAgainst_H_avg"
    ]
    away_features = [
        "GoalsScore_A", "GoalsAgainst_A", "GoalDifference_A",
        "WinStreak_A", "LoseStreak_A", "Wins_A", "Losses_A", "WinRate_A",
        "GoalsScore_A_avg", "GoalsAgainst_A_avg"
    ]

    df_baseline = df_baseline.merge(latest_home[home_features], left_on="HomeTeam", right_index=True, how="left")
    df_baseline = df_baseline.merge(latest_away[away_features], left_on="AwayTeam", right_index=True, how="left")

    # Feature engineering
    df_baseline["Elo_diff"] = df_baseline["Elo_H_before"] - df_baseline["Elo_A_before"]
    df_baseline["Elo_ratio"] = df_baseline["Elo_H_before"] / df_baseline["Elo_A_before"]
    df_baseline["Goals_likelyhood_H"] = df_baseline["GoalsScore_H_avg"] + df_baseline["GoalsAgainst_A_avg"]
    df_baseline["Goals_likelyhood_A"] = df_baseline["GoalsScore_A_avg"] + df_baseline["GoalsAgainst_H_avg"]
    df_baseline["Home_adv_elo_sum"] = df_baseline["Elo_H_before"] + df_baseline["Home_adv_elo"]

    # Encode
    team_cols = ["HomeTeam", "AwayTeam"]
    le = LabelEncoder()
    all_teams = pd.concat([df_baseline[col] for col in team_cols]).unique()
    le.fit(all_teams)
    for col in team_cols:
        df_baseline[col + "_code"] = le.transform(df_baseline[col])

    return df_baseline

# -------------------------
# Feature engineering 
# -------------------------
def feature_engineering_df(df: pd.DataFrame) -> pd.DataFrame:
    if {"Elo_H_before", "Elo_A_before"}.issubset(df.columns):
        df["Elo_diff"] = df["Elo_H_before"] - df["Elo_A_before"]
        with np.errstate(divide="ignore", invalid="ignore"):
            df["Elo_ratio"] = df["Elo_H_before"] / df["Elo_A_before"]
    if {"GoalsScore_H_avg", "GoalsAgainst_A_avg"}.issubset(df.columns):
        df["Goals_likelyhood_H"] = df["GoalsScore_H_avg"] + df["GoalsAgainst_A_avg"]
    if {"GoalsScore_A_avg", "GoalsAgainst_H_avg"}.issubset(df.columns):
        df["Goals_likelyhood_A"] = df["GoalsScore_A_avg"] + df["GoalsAgainst_H_avg"]
    if {"Elo_H_before", "Home_adv_elo"}.issubset(df.columns):
        df["Home_adv_elo_sum"] = df["Elo_H_before"] + df["Home_adv_elo"]
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



def _discover_model_pairs(models_dir: str) -> Dict[str, Dict[str, str]]:
    pairs: Dict[str, Dict[str, str]] = {}
    try:
        files = os.listdir(models_dir)
    except Exception:
        files = []

    # Legacy default pair
    if "home_model.pkl" in files and "away_model.pkl" in files:
        pairs["default"] = {
            "home": os.path.join(models_dir, "home_model.pkl"),
            "away": os.path.join(models_dir, "away_model.pkl"),
        }

    # Pattern: home_<name>_model.pkl and away_<name>_model.pkl
    for fname in files:
        if fname.startswith("home_") and fname.endswith("_model.pkl"):
            middle = fname[len("home_"):-len("_model.pkl")]
            away_name = f"away_{middle}_model.pkl"
            if away_name in files:
                pairs[middle] = {
                    "home": os.path.join(models_dir, fname),
                    "away": os.path.join(models_dir, away_name),
                }
    return pairs


@st.cache_resource(show_spinner=True)
def load_models_and_metrics(csv_path: str):
    df = read_dataset(csv_path)
    # Separate dataset for head-to-head lookup
    try:
        df_h2h = read_dataset(H2H_DATA_PATH)
    except Exception:
        df_h2h = df.copy()

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
        st.warning("Không có mapping team->code. Sẽ cố gắng suy luận từ dữ liệu, nhưng nên dump đúng từ notebook.")

    # Build fallback mapping from dataset if needed
    try:
        all_teams_series = pd.concat([
            df.get("HomeTeam", pd.Series(dtype=str)),
            df.get("AwayTeam", pd.Series(dtype=str)),
        ], axis=0).dropna().astype(str)
        unique_teams = sorted(all_teams_series.unique())
        inferred_map = {name: idx for idx, name in enumerate(unique_teams)}
    except Exception:
        inferred_map = {}

    # Fit StandardScaler on full dataset (feature-engineered), mirroring notebook
    df_fe = feature_engineering_df(df.copy())
    exclude_norm = {TARGET_HOME, TARGET_AWAY}
    cols_to_norm = (
        df_fe.select_dtypes(include=["int64", "float64"]).columns.difference(exclude_norm)
    )
    cols_to_norm = [c for c in cols_to_norm if c not in ["HomeTeam_code", "AwayTeam_code"]]
    scaler = StandardScaler()
    try:
        scaler.fit(df_fe.loc[:, cols_to_norm])
    except Exception:
        pass

    # Discover available model pairs and pick default
    model_pairs = _discover_model_pairs(os.path.join(os.getcwd(), "models"))
    selected_key = None
    # Preference order: 'rf' then 'default' then first available
    if "rf" in model_pairs:
        selected_key = "rf"
    elif "default" in model_pairs:
        selected_key = "default"
    elif model_pairs:
        selected_key = sorted(model_pairs.keys())[0]

    # Load selected models; fallback to configured paths
    try:
        if selected_key:
            model_home = load(model_pairs[selected_key]["home"])
            model_away = load(model_pairs[selected_key]["away"])
    except Exception:
        # keep previously loaded model_home/model_away
        pass

    return (
        df,
        df_h2h,
        model_home,
        model_away,
        feature_cols,
        (team_to_code or inferred_map),
        cols_to_norm,
        scaler,
        model_pairs,
        selected_key,
    )


def _mean_or_nan(series: pd.Series) -> float:
    series = series.dropna()
    if series.empty:
        return np.nan
    return float(series.mean())


def preprocessing(
    df: pd.DataFrame,
    home_team: str,
    away_team: str,
    feature_cols: List[str],
    team_to_code: Dict[str, int],
    cols_to_norm: List[str],
    scaler: Any | None,
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

    # Compute derived features if model expects them
    if "Elo_diff" in feature_cols and "Elo_diff" not in row:
        row["Elo_diff"] = float(row.get("Elo_H_before", 0.0)) - float(row.get("Elo_A_before", 0.0))
    if "Elo_ratio" in feature_cols and "Elo_ratio" not in row:
        denom = float(row.get("Elo_A_before", 1.0)) or 1.0
        row["Elo_ratio"] = float(row.get("Elo_H_before", 0.0)) / denom
    if "Home_adv_elo_sum" in feature_cols and "Home_adv_elo_sum" not in row:
        row["Home_adv_elo_sum"] = float(row.get("Home_adv_elo", 0.0))
    if "Goals_likelyhood_H" in feature_cols and "Goals_likelyhood_H" not in row:
        row["Goals_likelyhood_H"] = float(np.nanmean([
            row.get("GoalsScore_H_avg", np.nan),
            row.get("GoalsAgainst_A_avg", np.nan),
        ])) if not (np.isnan(row.get("GoalsScore_H_avg", np.nan)) and np.isnan(row.get("GoalsAgainst_A_avg", np.nan))) else 0.0
    if "Goals_likelyhood_A" in feature_cols and "Goals_likelyhood_A" not in row:
        row["Goals_likelyhood_A"] = float(np.nanmean([
            row.get("GoalsScore_A_avg", np.nan),
            row.get("GoalsAgainst_H_avg", np.nan),
        ])) if not (np.isnan(row.get("GoalsScore_A_avg", np.nan)) and np.isnan(row.get("GoalsAgainst_H_avg", np.nan))) else 0.0

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
    # Fill NaNs: if codes missing, don't silently zero – try fallback based on overall index
    if np.isnan(row_df["HomeTeam_code"].iloc[0]) or np.isnan(row_df["AwayTeam_code"].iloc[0]):
        st.warning("Thiếu mã đội (code) cho một trong hai đội. Đang dùng suy luận tạm thời.")
    row_df = row_df.fillna(row_df.median(numeric_only=True)).fillna(0.0)
    # Apply feature engineering to the single-row frame to ensure derived cols exist
    row_df = feature_engineering_df(row_df)

    # Apply normalization via StandardScaler
    if scaler is not None and cols_to_norm:
        try:
            row_df.loc[:, cols_to_norm] = scaler.transform(row_df.loc[:, cols_to_norm])
        except Exception:
            pass

    return row_df

def show_stats(
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

    # Compute derived features if model expects them
    if "Elo_diff" in feature_cols and "Elo_diff" not in row:
        row["Elo_diff"] = float(row.get("Elo_H_before", 0.0)) - float(row.get("Elo_A_before", 0.0))
    if "Elo_ratio" in feature_cols and "Elo_ratio" not in row:
        denom = float(row.get("Elo_A_before", 1.0)) or 1.0
        row["Elo_ratio"] = float(row.get("Elo_H_before", 0.0)) / denom
    if "Home_adv_elo_sum" in feature_cols and "Home_adv_elo_sum" not in row:
        row["Home_adv_elo_sum"] = float(row.get("Home_adv_elo", 0.0))
    if "Goals_likelyhood_H" in feature_cols and "Goals_likelyhood_H" not in row:
        row["Goals_likelyhood_H"] = float(np.nanmean([
            row.get("GoalsScore_H_avg", np.nan),
            row.get("GoalsAgainst_A_avg", np.nan),
        ])) if not (np.isnan(row.get("GoalsScore_H_avg", np.nan)) and np.isnan(row.get("GoalsAgainst_A_avg", np.nan))) else 0.0
    if "Goals_likelyhood_A" in feature_cols and "Goals_likelyhood_A" not in row:
        row["Goals_likelyhood_A"] = float(np.nanmean([
            row.get("GoalsScore_A_avg", np.nan),
            row.get("GoalsAgainst_H_avg", np.nan),
        ])) if not (np.isnan(row.get("GoalsScore_A_avg", np.nan)) and np.isnan(row.get("GoalsAgainst_H_avg", np.nan))) else 0.0

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
    # Fill NaNs: if codes missing, don't silently zero – try fallback based on overall index
    if np.isnan(row_df["HomeTeam_code"].iloc[0]) or np.isnan(row_df["AwayTeam_code"].iloc[0]):
        st.warning("Thiếu mã đội (code) cho một trong hai đội. Đang dùng suy luận tạm thời.")
    row_df = row_df.fillna(row_df.median(numeric_only=True)).fillna(0.0)
    # Apply feature engineering to the single-row frame to ensure derived cols exist
    row_df = feature_engineering_df(row_df)


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
st.title("⚽ Dự đoán tỉ số bóng đá")
st.caption("Dự đoán tỉ số cho cặp đấu được chọn từ dữ liệu lịch sử.")

data_path = DEFAULT_DATA_PATH

if "_cache_models" not in st.session_state:
    with st.spinner("Đang tải mô hình đã huấn luyện..."):
        (
            df,
            df_h2h,
            model_home,
            model_away,
            feature_cols,
            team_to_code,
            cols_to_norm,
            scaler,
            model_pairs,
            selected_model_key,
        ) = load_models_and_metrics(data_path)
        st.session_state["_cache_models"] = (
            df,
            df_h2h,
            model_home,
            model_away,
            feature_cols,
            team_to_code,
            cols_to_norm,
            scaler,
            model_pairs,
            selected_model_key,
        )

(
    df,
    df_h2h,
    model_home,
    model_away,
    feature_cols,
    team_to_code,
    cols_to_norm,
    scaler,
    model_pairs,
    selected_model_key,
) = st.session_state.get("_cache_models", load_models_and_metrics(data_path))

# Teams list for selection
df_test = df.copy()
teams_sorted = sorted(
    set(df_test.get("HomeTeam", pd.Series(dtype=str)).dropna().unique())
    | set(df_test.get("AwayTeam", pd.Series(dtype=str)).dropna().unique())
)

main_col = st.container()
with main_col:
    st.subheader("Chọn mô hình và cặp đấu")
    # Model picker
    model_keys = sorted(model_pairs.keys())
    if not model_keys:
        st.error("Không tìm thấy mô hình trong thư mục models/.")
        model_choice = None
    else:
        default_index = model_keys.index(selected_model_key) if selected_model_key in model_keys else 0
        model_choice = st.selectbox("Mô hình", options=model_keys, index=default_index, key="model_select")
        # If user changed model, reload models
        prev_model_choice = st.session_state.get("_model_choice")
        if model_choice and model_choice != prev_model_choice:
            try:
                model_home = load(model_pairs[model_choice]["home"])
                model_away = load(model_pairs[model_choice]["away"])
                st.session_state["_model_choice"] = model_choice
                # update cache tuple
                st.session_state["_cache_models"] = (
                    df, df_h2h, model_home, model_away, feature_cols, team_to_code, cols_to_norm, scaler, model_pairs, model_choice
                )
            except Exception as _e:
                st.warning(f"Không thể tải mô hình '{model_choice}'. Dùng mô hình trước đó.")
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
        # If custom df_final is present and enabled, use only the selected matchup from it; else use regular preprocessing
        if st.session_state.get("_use_custom_h2h", False) and st.session_state.get("_df_final_custom") is not None:
            df_final_custom = st.session_state.get("_df_final_custom")
            try:
                sel = df_final_custom[(df_final_custom["HomeTeam"] == home_team) & (df_final_custom["AwayTeam"] == away_team)]
                if sel.empty:
                    st.error("Custom dataset không có cặp đấu đã chọn. Đang dùng dự đoán mặc định.")
                    raise ValueError("Selected matchup not in custom df")
                X_pred = sel[feature_cols].copy()
                try:
                    X_pred.loc[:, cols_to_norm] = scaler.transform(X_pred.loc[:, cols_to_norm])
                except Exception:
                    pass
                # Ensure codes numeric for debug consistency
                if "HomeTeam_code" in X_pred.columns:
                    X_pred["HomeTeam_code"] = pd.to_numeric(X_pred["HomeTeam_code"], errors="coerce").fillna(0).astype(int)
                if "AwayTeam_code" in X_pred.columns:
                    X_pred["AwayTeam_code"] = pd.to_numeric(X_pred["AwayTeam_code"], errors="coerce").fillna(0).astype(int)
                y_home_pred = model_home.predict(X_pred)
                y_away_pred = model_away.predict(X_pred)
                raw_home = float(y_home_pred[0] if not hasattr(y_home_pred, 'iloc') else y_home_pred.iloc[0])
                raw_away = float(y_away_pred[0] if not hasattr(y_away_pred, 'iloc') else y_away_pred.iloc[0])
                st.session_state["_raw_home"] = raw_home
                st.session_state["_raw_away"] = raw_away
                feats_row = X_pred.iloc[0].to_dict()
                st.session_state["_last_features_row"] = feats_row
                st.session_state["_score_home"] = int(np.rint(raw_home))
                st.session_state["_score_away"] = int(np.rint(raw_away))
                st.success("Đã dùng custom dataset cho dự đoán.")
            except Exception as e:
                st.error("Lỗi dùng custom dataset, chuyển sang dự đoán mặc định.")
                st.exception(e)
                X_row = preprocessing(df, home_team, away_team, feature_cols, team_to_code, cols_to_norm, scaler)
                g_home, g_away = predict_score(model_home, model_away, X_row)
                # store raw for debug consistency
                try:
                    st.session_state["_raw_home"] = float(model_home.predict(X_row)[0])
                    st.session_state["_raw_away"] = float(model_away.predict(X_row)[0])
                    st.session_state["_last_features_row"] = X_row.iloc[0].to_dict()
                except Exception:
                    pass
                st.session_state["_score_home"] = g_home
                st.session_state["_score_away"] = g_away
        else:
            X_row = preprocessing(df, home_team, away_team, feature_cols, team_to_code, cols_to_norm, scaler)
            g_home, g_away = predict_score(model_home, model_away, X_row)
            try:
                st.session_state["_raw_home"] = float(model_home.predict(X_row)[0])
                st.session_state["_raw_away"] = float(model_away.predict(X_row)[0])
                st.session_state["_last_features_row"] = X_row.iloc[0].to_dict()
            except Exception:
                pass
            st.session_state["_score_home"] = g_home
            st.session_state["_score_away"] = g_away

    disp_home = st.session_state.get("_score_home")
    disp_away = st.session_state.get("_score_away")
    disp_home = disp_home if disp_home is not None else "?"
    disp_away = disp_away if disp_away is not None else "?"

    if home_team and away_team and home_team != away_team:
        render_scoreboard(home_team, away_team, disp_home, disp_away)

        # Optional debug: show raw predictions
        with st.expander("Chi tiết dự đoán"):
            try:
                raw_home = st.session_state.get("_raw_home")
                raw_away = st.session_state.get("_raw_away")
                feats = st.session_state.get("_last_features_row", {})
                if raw_home is None or raw_away is None:
                    st.write("Chọn dự đoán để xem")
                else:
                    out = {
                        "raw_home_pred": round(float(raw_home), 4),
                        "raw_away_pred": round(float(raw_away), 4),
                        "rounded_home": int(np.rint(float(raw_home))),
                        "rounded_away": int(np.rint(float(raw_away))),
                    }
                    if feats:
                        if "HomeTeam_code" in feats:
                            out["HomeTeam_code"] = int(feats["HomeTeam_code"]) if feats["HomeTeam_code"] is not None else None
                        if "AwayTeam_code" in feats:
                            out["AwayTeam_code"] = int(feats["AwayTeam_code"]) if feats["AwayTeam_code"] is not None else None
                    st.write(out)
            except Exception:
                st.write("Chọn dự đoán để xem")
    # Head-to-head last 5 from data.csv
    if home_team and away_team and home_team != away_team:
        try:
            mask_pair = (
                ((df_h2h.get("HomeTeam") == home_team) & (df_h2h.get("AwayTeam") == away_team))
                |
                ((df_h2h.get("HomeTeam") == away_team) & (df_h2h.get("AwayTeam") == home_team))
            )
            cols = ["Date", "HomeTeam", "AwayTeam", TARGET_HOME, TARGET_AWAY]
            cols = [c for c in cols if c in df_h2h.columns]
            h2h = df_h2h.loc[mask_pair, cols].dropna()
            if "Date" in h2h.columns:
                h2h = h2h.sort_values("Date", ascending=False)
                try:
                    h2h["Date"] = pd.to_datetime(h2h["Date"], errors="coerce").dt.date.astype(str)
                except Exception:
                    pass
            h2h = h2h.head(5).rename(columns={TARGET_HOME: "FT_Home", TARGET_AWAY: "FT_Away"})
            st.subheader("Đối đầu gần nhất (tối đa 5 trận)")
            st.dataframe(h2h.reset_index(drop=True), use_container_width=True)
            # Custom H2H input below the table
            with st.container():
                st.subheader("Custome dataset")
                st.caption("Tùy chỉnh tỉ số đối đầu (số trận tùy ý)")
                st.checkbox(
                    "Sử dụng custom dataset cho dự đoán",
                    value=bool(st.session_state.get("_use_custom_h2h", False)),
                    key="_use_custom_h2h",
                )
                # Team options from dataset, excluding the currently selected home/away
                all_teams = sorted(
                    set(df.get("HomeTeam", pd.Series(dtype=str)).dropna().unique())
                    | set(df.get("AwayTeam", pd.Series(dtype=str)).dropna().unique())
                )
                team_options_home = [t for t in all_teams if t != home_team]
                team_options_away = [t for t in all_teams if t != away_team]

                # Safe updater for count per panel
                def _inc_count(prefix: str, max_rows: int = 5):
                    key = f"{prefix}_count"
                    cur = int(st.session_state.get(key, 1))
                    if cur < max_rows:
                        st.session_state[key] = cur + 1

                # Delete a specific row (by index) for a given panel
                def _delete_row(prefix: str, idx: int):
                    count_key = f"{prefix}_count"
                    cur = int(st.session_state.get(count_key, 1))
                    if cur <= 1:
                        return
                    keep_indices = [j for j in range(cur) if j != idx]
                    # Repack values we keep
                    teams = [st.session_state.get(f"{prefix}_team_{j}") for j in keep_indices]
                    venues = [st.session_state.get(f"{prefix}_venue_{j}") for j in keep_indices]
                    gfs = [st.session_state.get(f"{prefix}_gf_{j}") for j in keep_indices]
                    gas = [st.session_state.get(f"{prefix}_ga_{j}") for j in keep_indices]
                    new_count = max(1, cur - 1)
                    # Write back compacted values
                    for i2 in range(new_count):
                        st.session_state[f"{prefix}_team_{i2}"] = teams[i2] if i2 < len(teams) else None
                        st.session_state[f"{prefix}_venue_{i2}"] = venues[i2] if i2 < len(venues) else "Home"
                        st.session_state[f"{prefix}_gf_{i2}"] = int(gfs[i2]) if i2 < len(gfs) and gfs[i2] is not None else 0
                        st.session_state[f"{prefix}_ga_{i2}"] = int(gas[i2]) if i2 < len(gas) and gas[i2] is not None else 0
                    # Clean up old keys
                    for j in range(new_count, cur):
                        for field in ("team", "venue", "gf", "ga"):
                            k = f"{prefix}_{field}_{j}"
                            if k in st.session_state:
                                del st.session_state[k]
                    st.session_state[count_key] = new_count

                def render_custom_inputs(panel_title: str, team_options: List[str], default_venue: str, key_prefix: str, is_home_panel: bool):
                    st.subheader(panel_title)
                    st.write("Mỗi trận: chọn Đội, chọn Sân (Home/Away), nhập GF/GA")
                    count_key = f"{key_prefix}_count"
                    # Always start with 1 if no state; cap to max 5
                    cur_count = st.session_state.get(count_key, 1)
                    max_rows = 5
                    if cur_count > max_rows:
                        cur_count = max_rows
                        st.session_state[count_key] = max_rows
                    # Render rows
                    for i in range(cur_count):
                        # Read current values if any; DO NOT write to session_state before widget creation
                        team_val = st.session_state.get(
                            f"{key_prefix}_team_{i}", team_options[0] if len(team_options) > 0 else None
                        )
                        venue_val = st.session_state.get(f"{key_prefix}_venue_{i}", default_venue)
                        gf_val = int(st.session_state.get(f"{key_prefix}_gf_{i}", 0))
                        ga_val = int(st.session_state.get(f"{key_prefix}_ga_{i}", 0))
                        c0, c1, c2, c3, c4, c5 = st.columns([0.6, 1.6, 1.2, 1.0, 1.0, 0.6])
                        with c0:
                            st.markdown(f"Trận {i+1}")
                        with c1:
                            st.selectbox(
                                "Đội",
                                options=team_options,
                                index=(team_options.index(team_val) if team_options and team_val in team_options else (0 if team_options else 0)),
                                key=f"{key_prefix}_team_{i}",
                            )
                        with c2:
                            st.selectbox(
                                "Sân (đội đang xét)",
                                options=["Home", "Away"],
                                index=0 if venue_val == "Home" else 1,
                                key=f"{key_prefix}_venue_{i}",
                            )
                        with c3:
                            st.number_input(
                                "GF",
                                min_value=0,
                                max_value=15,
                                value=gf_val,
                                key=f"{key_prefix}_gf_{i}",
                            )
                        with c4:
                            st.number_input(
                                "GA",
                                min_value=0,
                                max_value=15,
                                value=ga_val,
                                key=f"{key_prefix}_ga_{i}",
                            )
                        with c5:
                            st.button(
                                "✕",
                                key=f"del_{key_prefix}_{i}",
                                disabled=(cur_count <= 1),
                                on_click=_delete_row,
                                args=(key_prefix, i),
                            )
                    # Add row button (per column) with max 5 rows
                    add_disabled = cur_count >= max_rows
                    st.button(
                        "+ Thêm trận",
                        key=f"add_{key_prefix}",
                        disabled=add_disabled,
                        on_click=_inc_count,
                        args=(key_prefix, max_rows),
                    )
                    if add_disabled:
                        st.caption("Đã đạt tối đa 5 trận.")
                        
                col_left, col_right = st.columns(2)
                with col_left:
                    render_custom_inputs("Đội nhà ", team_options_home, "Home", "custom_home", True)
                with col_right:
                    render_custom_inputs("Đội khách", team_options_away, "Away", "custom_away", False)

                # Build new_data from custom inputs if enabled
                if st.session_state.get("_use_custom_h2h", False):
                    def _collect(prefix: str, is_home: bool) -> List[Dict[str, Any]]:
                        rows: List[Dict[str, Any]] = []
                        cnt = int(st.session_state.get(f"{prefix}_count", 1))
                        for i in range(cnt):
                            team_sel = st.session_state.get(f"{prefix}_team_{i}")
                            venue = st.session_state.get(f"{prefix}_venue_{i}", "Home")
                            gf = int(st.session_state.get(f"{prefix}_gf_{i}", 0))
                            ga = int(st.session_state.get(f"{prefix}_ga_{i}", 0))
                            if team_sel is None:
                                continue
                            if is_home:
                                # HomeTeam: current home_team; AwayTeam: selected opponent
                                rows.append({
                                    "HomeTeam": home_team,
                                    "AwayTeam": team_sel,
                                    "Full Time Home Goals": gf if venue == "Home" else ga,
                                    "Full Time Away Goals": ga if venue == "Home" else gf,
                                })
                            else:
                                # Away panel: venue is for current away_team (đội đang xét)
                                if venue == "Home":
                                    # away_team plays at Home
                                    rows.append({
                                        "HomeTeam": away_team,
                                        "AwayTeam": team_sel,
                                        "Full Time Home Goals": gf,
                                        "Full Time Away Goals": ga,
                                    })
                                else:
                                    # away_team plays Away
                                    rows.append({
                                        "HomeTeam": team_sel,
                                        "AwayTeam": away_team,
                                        "Full Time Home Goals": ga,
                                        "Full Time Away Goals": gf,
                                    })
                        return rows

                    new_rows_home = _collect("custom_home", True)
                    new_rows_away = _collect("custom_away", False)
                    new_rows = new_rows_home + new_rows_away
                    if new_rows:
                        # Prepare features and persist for main prediction button
                        try:
                            df_final = prepare_features_for_prediction(
                                folder="data_season",
                                new_data=new_rows,
                                baseline_path="./data_extra/new_season.csv",
                            )
                            st.session_state["_df_final_custom"] = df_final.copy()
                            st.success("Custom dataset đã sẵn sàng để dự đoán bằng nút bên trên.")
                        except Exception as e:
                            st.error("Không thể tạo đặc trưng từ custom dataset.")
                            st.exception(e)
        except Exception as e:
            st.info("Không lấy được lịch sử đối đầu.")
            st.exception(e)  
   

    # Separator
    st.markdown("---")


    with st.expander("Ghi chú & Hướng dẫn"):
        st.markdown(
            """
            - Ứng dụng sử dụng 2 mô hình XGBoost Regression để dự đoán số bàn của đội nhà và đội khách.
            - Đặc trưng đầu vào gồm: one-hot của `HomeTeam`/`AwayTeam` và các cột số trong dữ liệu (ví dụ Elo, chuỗi trận, trung bình bàn thắng/thua,...).
            - Khi dự đoán cho cặp đấu mới, các đặc trưng số được ước lượng bằng trung bình có trọng số theo vai trò (Home/ Away) và đối đầu trực tiếp nếu có.
            - Các kết quả chỉ mang tính tham khảo, không dùng cho mục đích cá cược.
            """
        )

    

