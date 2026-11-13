import json
import math
import random
from collections import deque
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

# ─────────────────────────── 기본 설정 ───────────────────────────
APP_TITLE = "움직임 분류"
APP_ICON = "🤖"
st.set_page_config(page_title=APP_TITLE, page_icon=APP_ICON, layout="wide")

DATA_URL = "https://raw.githubusercontent.com/TENDO-Sj/data/refs/heads/main/Robo.csv"
SOURCE_URL = "https://www.kaggle.com/datasets/jimschacko/wall-following-robot-navigation-dataset/data"

RECORD_DIR = Path(__file__).parent / "내_기록"
RECORD_FILE = RECORD_DIR / "열_설명.json"

SENSORS = [f"V{i}" for i in range(1, 25)]


def sensor_angle(i: int) -> int:
    """V1=180°, V2~V12 = -165°~-15°, V13~V24 = 0°~165°"""
    if i == 1:
        return 180
    if i <= 12:
        return -165 + 15 * (i - 2)
    return 15 * (i - 13)


ANGLES = {f"V{i}": sensor_angle(i) for i in range(1, 25)}

CLASS_EN = {1: "Move-Forward", 2: "Slight-Right-Turn", 3: "Sharp-Right-Turn", 4: "Slight-Left-Turn"}
CLASS_KO = {1: "직진", 2: "약간 우회전", 3: "급격한 우회전", 4: "약간 좌회전"}


def class_label(c: int) -> str:
    return f"{c}: {CLASS_KO[c]} ({CLASS_EN[c]})"


def default_meaning(col: str) -> str:
    if col == "id":
        return "행 번호"
    if col == "Class":
        return "로봇의 움직임 (1 직진, 2 약간 우회전, 3 급격한 우회전, 4 약간 좌회전)"
    a = ANGLES[col]
    if col == "V1":
        return f"로봇 전면의 초음파 센서 판독값 (기준 각도: {a}°)"
    if col == "V13":
        return f"로봇 후면의 초음파 센서 판독값 (기준 각도: {a}°)"
    return f"초음파 센서 판독값 (기준 각도: {a}°)"


ALL_COLS = ["id"] + SENSORS + ["Class"]


# ─────────────────────────── 데이터 ───────────────────────────
@st.cache_data(show_spinner="데이터를 불러오는 중...")
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_URL, encoding="utf-8")
    df.columns = [c.strip() for c in df.columns]
    # Class가 글자로 들어 있으면 숫자로 바꿈
    if df["Class"].dtype == object:
        name_to_num = {v.lower(): k for k, v in CLASS_EN.items()}
        df["Class"] = (
            df["Class"].astype(str).str.strip().str.lower().map(name_to_num).fillna(df["Class"])
        )
    df["Class"] = pd.to_numeric(df["Class"], errors="coerce").astype("Int64")
    return df


# ─────────────────────────── 기록(JSON) ───────────────────────────
def load_records() -> pd.DataFrame:
    saved = {}
    if RECORD_FILE.exists():
        try:
            for row in json.loads(RECORD_FILE.read_text(encoding="utf-8")):
                saved[row["열 이름"]] = row
        except (json.JSONDecodeError, KeyError, TypeError):
            saved = {}
    rows = []
    for col in ALL_COLS:
        r = saved.get(col, {})
        rows.append(
            {
                "열 이름": col,
                "우리말 뜻": r.get("우리말 뜻", default_meaning(col)),
                "출처": r.get("출처", SOURCE_URL),
            }
        )
    return pd.DataFrame(rows)


def save_records(df: pd.DataFrame) -> None:
    RECORD_DIR.mkdir(parents=True, exist_ok=True)
    data = df.fillna("").to_dict(orient="records")
    RECORD_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


# ─────────────────────────── 전처리 ───────────────────────────
OUTLIER_MODES = ["그대로 두기", "제거", "경계값으로 자르기"]
SCALERS = {
    "사용 안 함": None,
    "Min-Max (0~1)": MinMaxScaler,
    "표준화 (평균 0, 표준편차 1)": StandardScaler,
    "Robust (중앙값·IQR 기준)": RobustScaler,
}
DEFAULT_PREP = {"drop_dup": False, "outlier": OUTLIER_MODES[0], "k": 1.5}


def get_prep() -> dict:
    """전처리 화면에서 고른 설정 (아직 안 골랐으면 기본값)"""
    return {**DEFAULT_PREP, **st.session_state.get("prep", {})}


def iqr_bounds(data: pd.DataFrame, k: float):
    q1, q3 = data[SENSORS].quantile(0.25), data[SENSORS].quantile(0.75)
    iqr = q3 - q1
    return q1 - k * iqr, q3 + k * iqr


def clean_basic(data: pd.DataFrame) -> pd.DataFrame:
    """Class·센서값이 비어 있는 행은 항상 뺀다"""
    return data.dropna(subset=SENSORS + ["Class"]).copy()


def preprocess(data: pd.DataFrame, drop_dup: bool, outlier: str, k: float) -> pd.DataFrame:
    out = clean_basic(data)
    if drop_dup:
        out = out.drop_duplicates(subset=SENSORS + ["Class"])
    if outlier != OUTLIER_MODES[0]:
        lo, hi = iqr_bounds(out, k)
        if outlier == OUTLIER_MODES[1]:
            mask = (out[SENSORS] < lo) | (out[SENSORS] > hi)
            out = out[~mask.any(axis=1)]
        else:
            out[SENSORS] = out[SENSORS].clip(lower=lo, upper=hi, axis=1)
    return out


# ─────────────────────────── 모델 ───────────────────────────
@st.cache_resource(show_spinner="모델을 학습하는 중...")
def train_model(test_size: float, n_trees: int, drop_dup: bool, outlier: str, k: float):
    df = preprocess(load_data(), drop_dup, outlier, k)
    X = df[SENSORS]
    y = df["Class"].astype(int)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    model = RandomForestClassifier(n_estimators=n_trees, random_state=42, n_jobs=-1)
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    return model, X_te, y_te, pred


def polar_chart(values: pd.Series, title: str) -> go.Figure:
    angles = [ANGLES[s] for s in SENSORS]
    fig = go.Figure(
        go.Barpolar(
            r=values.values,
            theta=angles,
            width=[14] * 24,
            text=SENSORS,
            hovertemplate="%{text}<br>각도 %{theta}°<br>거리 %{r:.3f}<extra></extra>",
            marker_color=values.values,
            marker_colorscale="Viridis",
        )
    )
    fig.update_layout(
        title=title,
        polar=dict(angularaxis=dict(rotation=270, direction="counterclockwise")),
        height=420,
        margin=dict(t=60, b=20, l=20, r=20),
    )
    return fig


# ─────────────────────────── 미로 시뮬레이션 ───────────────────────────
# 좌표: x = 열(오른쪽 +), y = 행(아래쪽 +). 화면에서 y축이 아래로 가므로
# 방향 각도는 "시계 방향이 +" 입니다. (0° = 오른쪽, 90° = 아래쪽)
# 센서 V1은 로봇 앞쪽, 번호가 커질수록 15°씩 돌아갑니다.
CLASS_COLORS = {1: "#2E86DE", 2: "#F39C12", 3: "#E74C3C", 4: "#27AE60"}


def make_maze(rows: int, cols: int, seed: int):
    """깊이 우선 탐색으로 만든 완전 미로. 1 = 벽, 0 = 길"""
    rng = random.Random(seed)
    H, W = 2 * rows + 1, 2 * cols + 1
    g = [[1] * W for _ in range(H)]
    g[1][1] = 0
    stack, seen = [(0, 0)], {(0, 0)}
    while stack:
        r, c = stack[-1]
        nb = [
            (r + dr, c + dc, dr, dc)
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))
            if 0 <= r + dr < rows and 0 <= c + dc < cols and (r + dr, c + dc) not in seen
        ]
        if not nb:
            stack.pop()
            continue
        nr, nc, dr, dc = rng.choice(nb)
        g[2 * r + 1 + dr][2 * c + 1 + dc] = 0
        g[2 * nr + 1][2 * nc + 1] = 0
        seen.add((nr, nc))
        stack.append((nr, nc))
    return tuple(tuple(row) for row in g)


def is_wall(grid, cx: int, cy: int) -> bool:
    if cy < 0 or cy >= len(grid) or cx < 0 or cx >= len(grid[0]):
        return True
    return grid[cy][cx] == 1


def ray_cast(grid, x: float, y: float, ang_deg: float, max_d: float) -> float:
    """격자 DDA 방식 광선 추적 → 벽까지 거리(칸 단위)"""
    a = math.radians(ang_deg)
    dx, dy = math.cos(a), math.sin(a)
    mx, my = math.floor(x), math.floor(y)
    inf = float("inf")
    if dx > 0:
        sx, tdx, tmx = 1, 1 / dx, (mx + 1 - x) / dx
    elif dx < 0:
        sx, tdx, tmx = -1, -1 / dx, (x - mx) / -dx
    else:
        sx, tdx, tmx = 0, inf, inf
    if dy > 0:
        sy, tdy, tmy = 1, 1 / dy, (my + 1 - y) / dy
    elif dy < 0:
        sy, tdy, tmy = -1, -1 / dy, (y - my) / -dy
    else:
        sy, tdy, tmy = 0, inf, inf
    while True:
        if tmx < tmy:
            t, mx, tmx = tmx, mx + sx, tmx + tdx
        else:
            t, my, tmy = tmy, my + sy, tmy + tdy
        if t > max_d:
            return max_d
        if is_wall(grid, mx, my):
            return t


def sense(grid, x, y, heading, unit_m, vmin, vmax, orient):
    """24개 초음파 값(미터)을 계산 — 데이터와 같은 범위로 자름"""
    out = []
    for i in range(24):
        d = ray_cast(grid, x, y, heading + orient * 15 * i, vmax / unit_m)
        out.append(round(min(max(d * unit_m, vmin), vmax), 3))
    return out


def bfs_path(grid, start, goal):
    prev = {start: None}
    q = deque([start])
    while q:
        cur = q.popleft()
        if cur == goal:
            break
        r, c = cur
        for nr, nc in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
            if (nr, nc) not in prev and not is_wall(grid, nc, nr):
                prev[(nr, nc)] = cur
                q.append((nr, nc))
    path, cur = [], goal
    while cur is not None:
        path.append(cur)
        cur = prev.get(cur)
    return path[::-1]


@st.cache_resource(show_spinner="시뮬레이션용 모델을 학습하는 중...")
def sim_model(drop_dup: bool, outlier: str, k: float):
    """앞서 있던 자료(Robo.csv)를 전처리한 뒤 전체로 학습한 모델"""
    d = preprocess(load_data(), drop_dup, outlier, k)
    m = RandomForestClassifier(n_estimators=60, random_state=42, n_jobs=1)
    m.fit(d[SENSORS], d["Class"].astype(int))
    return m


def predict_rows(model, readings_list):
    X = pd.DataFrame(readings_list, columns=SENSORS)
    proba = model.predict_proba(X)
    classes = [int(c) for c in model.classes_]
    full = []
    for p in proba:
        dct = dict(zip(classes, p))
        full.append([float(dct.get(c, 0.0)) for c in (1, 2, 3, 4)])
    return full


def maze_points(grid):
    H, W = len(grid), len(grid[0])
    return (1, 1), (H - 2, W - 2)


@st.cache_data(show_spinner="최단 경로를 따라 초음파 값을 계산하는 중...")
def simulate_path(grid, unit_m, vmin, vmax, orient, prep_key):
    start, goal = maze_points(grid)
    cells = bfs_path(grid, start, goal)
    pts = []
    for (r0, c0), (r1, c1) in zip(cells, cells[1:]):
        pts.append((c0 + 0.5, r0 + 0.5))
        pts.append(((c0 + c1) / 2 + 0.5, (r0 + r1) / 2 + 0.5))
    pts.append((cells[-1][1] + 0.5, cells[-1][0] + 0.5))

    headings = []
    for i, (x, y) in enumerate(pts):
        if i + 1 < len(pts):
            nx, ny = pts[i + 1]
            headings.append(math.degrees(math.atan2(ny - y, nx - x)))
        else:
            headings.append(headings[-1] if headings else 0.0)

    readings = [sense(grid, x, y, h, unit_m, vmin, vmax, orient) for (x, y), h in zip(pts, headings)]
    probs = predict_rows(sim_model(*prep_key), readings)

    rows = []
    n = len(pts)
    for i, ((x, y), h, rd, p) in enumerate(zip(pts, headings, readings, probs)):
        cls = 1 + max(range(4), key=lambda k: p[k])
        row = {"시점": i, "구분": "출발" if i == 0 else ("도착" if i == n - 1 else "앞으로 지날 위치"),
               "x": round(x, 2), "y": round(y, 2), "방향(°)": round(h % 360, 1)}
        row.update(dict(zip(SENSORS, rd)))
        row.update({"예측 Class": cls, "예측 움직임": CLASS_KO[cls]})
        row.update({f"P({c})": round(p[c - 1], 3) for c in (1, 2, 3, 4)})
        rows.append(row)
    return pd.DataFrame(rows)


DIRS = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # (dr, dc): 0=오른쪽 1=아래 2=왼쪽 3=위 (시계 방향 순서)


@st.cache_data(show_spinner="예측한 움직임대로 로봇을 움직이는 중...")
def simulate_model_drive(grid, unit_m, vmin, vmax, orient, max_moves, prep_key):
    """갈림길마다 모델이 예측한 확률로 방향을 고름.
    직진 점수 = P(1), 오른쪽 점수 = P(2)+P(3), 왼쪽 점수 = P(4).
    벽이 있는 칸으로는 절대 가지 않고, 이미 지나온 칸은 피하며, 막다른 길이면 되돌아감."""
    model = sim_model(*prep_key)
    start, goal = maze_points(grid)
    r, c = start
    d = 0 if not is_wall(grid, c + 1, r) else 1
    heading = d * 90.0
    visited, stack = {start}, [start]
    frames = []  # (x, y, heading, 실행 Class, 비고)

    def add(x, y, h, cls, note=""):
        frames.append((x, y, h, cls, note))

    moves, reached = 0, False
    while moves < max_moves:
        if (r, c) == goal:
            reached = True
            break
        x, y = c + 0.5, r + 0.5
        rd = sense(grid, x, y, heading, unit_m, vmin, vmax, orient)
        p = predict_rows(model, [rd])[0]
        options = {d: p[0], (d + 1) % 4: p[1] + p[2], (d + 3) % 4: p[3]}  # 직진/오른쪽/왼쪽
        ranked = sorted(options, key=lambda k: -options[k])
        ok = [k for k in ranked
              if not is_wall(grid, c + DIRS[k][1], r + DIRS[k][0])
              and (r + DIRS[k][0], c + DIRS[k][1]) not in visited]
        if ok:
            nd = ok[0]
            nr, nc = r + DIRS[nd][0], c + DIRS[nd][1]
            if nd == d:
                cls = 1
            elif nd == (d + 1) % 4:
                cls = 3 if p[2] > p[1] else 2
            else:
                cls = 4
            note = "" if nd == ranked[0] else "벽·지나온 길이라 다음 순위 선택"
            stack.append((nr, nc))
            visited.add((nr, nc))
        else:  # 막다른 길 → 왔던 칸으로 되돌아감
            stack.pop()
            if not stack:
                break
            nr, nc = stack[-1]
            nd = DIRS.index((nr - r, nc - c))
            cls, note = 3, "막다른 길 → 되돌아감"

        # 회전 장면
        turn = ((nd - d + 2) % 4 - 2) * 90  # -90, 0, 90, 180(=-180)
        if turn == -180:
            turn = 180
        n_turn = abs(turn) // 30
        for k in range(1, n_turn + 1):
            add(x, y, heading + turn * k / n_turn, cls, note)
        heading += turn
        # 앞으로 한 칸 이동 장면
        for k in range(1, 5):
            add(x + DIRS[nd][1] * k / 4, y + DIRS[nd][0] * k / 4, heading, cls, note)
        r, c, d = nr, nc, nd
        moves += 1

    frames.insert(0, (start[1] + 0.5, start[0] + 0.5, frames[0][2] if frames else heading,
                      frames[0][3] if frames else 1, "출발"))
    readings = [sense(grid, x, y, h, unit_m, vmin, vmax, orient) for x, y, h, _, _ in frames]
    probs = predict_rows(model, readings)

    rows = []
    for i, ((x, y, h, cls, note), rd, p) in enumerate(zip(frames, readings, probs)):
        pred = 1 + max(range(4), key=lambda k: p[k])
        row = {"시점": i, "x": round(x, 2), "y": round(y, 2), "방향(°)": round(h % 360, 1)}
        row.update(dict(zip(SENSORS, rd)))
        row.update({"예측 Class": pred, "예측 움직임": CLASS_KO[pred],
                    "실행 Class": cls, "실행 움직임": CLASS_KO[cls], "비고": note})
        row.update({f"P({k})": round(p[k - 1], 3) for k in (1, 2, 3, 4)})
        rows.append(row)
    return pd.DataFrame(rows), reached, moves


def maze_animation(grid, sim_df, unit_m, orient, title, cls_col):
    H, W = len(grid), len(grid[0])
    start, goal = maze_points(grid)
    fig = make_subplots(
        rows=2, cols=1, row_heights=[0.74, 0.26], vertical_spacing=0.07,
        subplot_titles=(title, "움직임 예측 확률 (Class 1~4)"),
    )
    # 0: 벽
    fig.add_trace(go.Heatmap(
        z=[list(r) for r in grid], x=[c + 0.5 for c in range(W)], y=[r + 0.5 for r in range(H)],
        colorscale=[[0, "#F7F7F7"], [1, "#34495E"]], showscale=False, hoverinfo="skip",
    ), row=1, col=1)
    # 1: 출발/도착
    fig.add_trace(go.Scatter(
        x=[start[1] + 0.5, goal[1] + 0.5], y=[start[0] + 0.5, goal[0] + 0.5],
        mode="markers+text", text=["출발", "도착"], textposition="middle right",
        marker=dict(size=16, color=["#27AE60", "#E74C3C"], symbol="square"),
        hoverinfo="skip", showlegend=False,
    ), row=1, col=1)

    labels = [f"{c}: {CLASS_KO[c]}" for c in (1, 2, 3, 4)]
    xs, ys = sim_df["x"].tolist(), sim_df["y"].tolist()

    def frame_traces(i):
        r = sim_df.iloc[i]
        rx, ry = [], []
        for k, s in enumerate(SENSORS):
            d = r[s] / unit_m
            a = math.radians(r["방향(°)"] + orient * 15 * k)
            rx += [r["x"], r["x"] + d * math.cos(a), None]
            ry += [r["y"], r["y"] + d * math.sin(a), None]
        cls = int(r[cls_col])
        probs = [r[f"P({c})"] for c in (1, 2, 3, 4)]
        colors = [CLASS_COLORS[c] if c == cls else "#BDC3C7" for c in (1, 2, 3, 4)]
        return [
            go.Scatter(x=xs[: i + 1], y=ys[: i + 1], mode="lines",
                       line=dict(color="#8E44AD", width=2), hoverinfo="skip", showlegend=False),
            go.Scatter(x=rx, y=ry, mode="lines", line=dict(color="rgba(52,152,219,0.35)", width=1),
                       hoverinfo="skip", showlegend=False),
            go.Scatter(x=[r["x"]], y=[r["y"]], mode="markers",
                       marker=dict(size=18, color=CLASS_COLORS[cls], symbol="triangle-up",
                                   angle=r["방향(°)"] + 90, line=dict(color="black", width=1)),
                       hovertext=f"시점 {int(r['시점'])} · {CLASS_KO[cls]}", hoverinfo="text",
                       showlegend=False),
            go.Bar(x=labels, y=probs, marker_color=colors, text=[f"{p:.2f}" for p in probs],
                   textposition="outside", showlegend=False, hoverinfo="x+y"),
        ]

    first = frame_traces(0)
    for k, tr in enumerate(first):
        fig.add_trace(tr, row=1 if k < 3 else 2, col=1)

    n = len(sim_df)
    fig.frames = [go.Frame(data=frame_traces(i), traces=[2, 3, 4, 5], name=str(i)) for i in range(n)]

    fig.update_xaxes(range=[0, W], showgrid=False, zeroline=False, showticklabels=False, row=1, col=1)
    fig.update_yaxes(range=[H, 0], showgrid=False, zeroline=False, showticklabels=False,
                     scaleanchor="x", scaleratio=1, row=1, col=1)
    fig.update_yaxes(range=[0, 1.15], title="확률", row=2, col=1)

    anim = dict(frame=dict(duration=70, redraw=True), transition=dict(duration=0), fromcurrent=True)
    fig.update_layout(
        height=900, margin=dict(t=60, b=40, l=10, r=10),
        updatemenus=[dict(
            type="buttons", direction="left", x=0, y=1.07, xanchor="left",
            buttons=[dict(label="▶ 재생", method="animate", args=[None, anim]),
                     dict(label="⏸ 멈춤", method="animate",
                          args=[[None], dict(mode="immediate", frame=dict(duration=0, redraw=False))])],
        )],
        sliders=[dict(
            active=0, y=-0.02, currentvalue=dict(prefix="시점: "),
            steps=[dict(label=str(i), method="animate",
                        args=[[str(i)], dict(mode="immediate", frame=dict(duration=0, redraw=True),
                                             transition=dict(duration=0))]) for i in range(n)],
        )],
    )
    return fig


def new_maze_seed():
    st.session_state.maze_seed = random.randint(0, 99999)


# ─────────────────────────── 화면 ───────────────────────────
st.title(f"{APP_ICON} {APP_TITLE}")

try:
    df = load_data()
except Exception as e:  # noqa: BLE001
    st.error(f"데이터를 불러오지 못했어요: {e}")
    st.stop()

page = st.sidebar.radio("화면", ["데이터 소개", "상관관계", "전처리", "움직임 분류", "미로 시뮬레이션"])
st.sidebar.caption(f"[데이터 출처 (Kaggle)]({SOURCE_URL})")

# 1) 데이터 소개 ------------------------------------------------------
if page == "데이터 소개":
    st.header("이 데이터는 무엇인가요?")
    st.markdown(
        """
**벽 따라가기 로봇 주행 데이터(Wall-Following Robot Navigation)** 입니다.

로봇 몸체를 빙 둘러 **24개의 초음파 센서(V1~V24)** 가 15°마다 달려 있고,
각 센서는 벽까지의 거리를 잽니다. 로봇이 방 안에서 벽을 따라 시계 방향으로 돌 때,
그 순간의 센서값과 로봇이 실제로 한 **움직임(Class)** 이 함께 기록되어 있습니다.

| Class | 움직임 |
|---|---|
| 1 | 직진 (Move-Forward) |
| 2 | 약간 우회전 (Slight-Right-Turn) |
| 3 | 급격한 우회전 (Sharp-Right-Turn) |
| 4 | 약간 좌회전 (Slight-Left-Turn) |

목표는 **센서값 24개만 보고 로봇이 어떤 움직임을 할지 맞히는 것**입니다.
        """
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("행 수", f"{len(df):,}")
    c2.metric("센서 수", len(SENSORS))
    c3.metric("움직임 종류", df["Class"].nunique())

    st.subheader("처음 다섯 줄")
    st.dataframe(df.head(5), hide_index=True, width="stretch")

    st.subheader("움직임별 데이터 개수")
    counts = df["Class"].value_counts().sort_index().reset_index()
    counts.columns = ["Class", "개수"]
    counts["움직임"] = counts["Class"].map(lambda c: class_label(int(c)))
    fig = px.bar(counts, x="움직임", y="개수", color="움직임", text="개수")
    fig.update_layout(showlegend=False, height=380)
    st.plotly_chart(fig, width="stretch")

    st.subheader("열 설명 (우리말 뜻 · 출처)")
    st.caption("칸을 고치면 `내_기록/열_설명.json` 에 바로 저장되고, 새로고침해도 그대로 남아요.")

    if "records" not in st.session_state:
        st.session_state.records = load_records()

    edited = st.data_editor(
        st.session_state.records,
        key="record_editor",
        hide_index=True,
        disabled=["열 이름"],
        width="stretch",
        column_config={
            "열 이름": st.column_config.TextColumn(width="small"),
            "우리말 뜻": st.column_config.TextColumn(width="large"),
            "출처": st.column_config.TextColumn(width="medium"),
        },
    )
    if not edited.equals(st.session_state.get("last_saved", st.session_state.records)):
        save_records(edited)
        st.session_state.last_saved = edited.copy()
        st.toast("저장했어요 💾")

    if st.button("기본값으로 되돌리기"):
        if RECORD_FILE.exists():
            RECORD_FILE.unlink()
        for k in ("records", "last_saved", "record_editor"):
            st.session_state.pop(k, None)
        st.rerun()

# 2) 상관관계 ---------------------------------------------------------
elif page == "상관관계":
    st.header("상관관계표")
    include_id = st.checkbox("id 열도 포함", value=False)
    cols = (["id"] if include_id else []) + SENSORS + ["Class"]
    corr = df[cols].astype(float).corr()

    fig = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        aspect="auto",
    )
    fig.update_layout(height=800, margin=dict(t=30, b=10))
    fig.update_traces(textfont_size=9)
    st.plotly_chart(fig, width="stretch")

    with st.expander("숫자 표로 보기"):
        st.dataframe(corr.round(3), width="stretch")

    st.subheader("Class와의 상관계수")
    cc = corr["Class"].drop("Class").reset_index()
    cc.columns = ["열", "상관계수"]
    cc["설명"] = cc["열"].map(lambda c: f"{ANGLES[c]}°" if c in ANGLES else c)
    fig2 = px.bar(
        cc, x="열", y="상관계수", color="상관계수", hover_data=["설명"],
        color_continuous_scale="RdBu_r", range_color=[-1, 1],
    )
    fig2.update_layout(height=380)
    st.plotly_chart(fig2, width="stretch")

# 3) 전처리 -----------------------------------------------------------
elif page == "전처리":
    st.header("전처리")
    st.caption("여기서 고른 설정은 '움직임 분류' 학습에 그대로 적용돼요.")
    prep = get_prep()
    base_df = clean_basic(df)
    X_all = base_df[SENSORS]

    # 1. 결측치 · 중복
    st.subheader("1. 결측치와 중복")
    na = df[SENSORS + ["Class"]].isna().sum()
    n_dup = int(base_df.duplicated(subset=SENSORS + ["Class"]).sum())
    m1, m2, m3 = st.columns(3)
    m1.metric("결측치 칸 수", int(na.sum()))
    m2.metric("결측치가 있는 행", len(df) - len(base_df))
    m3.metric("중복 행", n_dup)
    if na.sum() > 0:
        st.dataframe(na[na > 0].rename("결측 개수"), width="stretch")
        st.info("결측치가 있는 행은 항상 제거해요.")
    else:
        st.success("결측치가 없어요.")
    drop_dup = st.checkbox("중복 행 제거", value=prep["drop_dup"])

    # 2. 이상치
    st.subheader("2. 이상치 (IQR 방식)")
    st.caption("Q1 − k×IQR 보다 작거나 Q3 + k×IQR 보다 큰 값을 이상치로 봐요.")
    c1, c2 = st.columns(2)
    k = c1.slider("k (클수록 덜 엄격)", 1.0, 3.0, float(prep["k"]), 0.5)
    outlier = c2.radio("처리 방법", OUTLIER_MODES, index=OUTLIER_MODES.index(prep["outlier"]),
                       horizontal=True)

    lo, hi = iqr_bounds(X_all, k)
    is_out = (X_all < lo) | (X_all > hi)
    o1, o2 = st.columns(2)
    o1.metric("이상치가 있는 행", f"{int(is_out.any(axis=1).sum()):,}")
    o2.metric("이상치 칸 수", f"{int(is_out.values.sum()):,}")

    cnt = is_out.sum().reset_index()
    cnt.columns = ["센서", "이상치 개수"]
    cnt["각도"] = cnt["센서"].map(ANGLES)
    fig = px.bar(cnt, x="센서", y="이상치 개수", hover_data=["각도"], title="센서별 이상치 개수")
    fig.update_layout(height=340)
    st.plotly_chart(fig, width="stretch")

    # 3. 스케일링 미리보기
    st.subheader("3. 스케일링 미리보기")
    st.caption("랜덤 포레스트는 값의 크기에 영향을 받지 않아서 학습에는 적용하지 않고, 분포만 비교해요.")
    scaler_name = st.selectbox("스케일러", list(SCALERS))
    picks = st.multiselect("볼 센서", SENSORS, default=["V1", "V7", "V13", "V19"])
    if picks:
        before = X_all[picks]
        scaler_cls = SCALERS[scaler_name]
        after = (
            pd.DataFrame(scaler_cls().fit_transform(before), columns=picks, index=before.index)
            if scaler_cls else before
        )
        left, right = st.columns(2)
        for box, data, title in ((left, before, "전"), (right, after, "후")):
            f = px.box(data.melt(var_name="센서", value_name="값"), x="센서", y="값", title=title)
            f.update_layout(height=360)
            box.plotly_chart(f, width="stretch")
        st.dataframe(after.describe().T.round(3), width="stretch")

    # 4. 결과 요약
    st.subheader("4. 전처리 결과")
    result = preprocess(df, drop_dup, outlier, k)
    r1, r2, r3 = st.columns(3)
    r1.metric("처음 행 수", f"{len(df):,}")
    r2.metric("전처리 후 행 수", f"{len(result):,}", delta=f"{len(result) - len(df):,}")
    r3.metric("남은 비율", f"{len(result) / len(df) * 100:.1f}%")

    cmp = pd.DataFrame({
        "전": df["Class"].value_counts().sort_index(),
        "후": result["Class"].value_counts().sort_index(),
    }).fillna(0).astype(int)
    cmp.index = [class_label(int(c)) for c in cmp.index]
    st.dataframe(cmp, width="stretch")

    st.session_state["prep"] = {"drop_dup": drop_dup, "outlier": outlier, "k": k}

# 4) 움직임 분류 ------------------------------------------------------
elif page == "움직임 분류":
    st.header("센서값으로 움직임 맞히기")
    prep = get_prep()
    st.caption(
        f"전처리: 중복 제거 {'O' if prep['drop_dup'] else 'X'} · 이상치 {prep['outlier']}"
        + ("" if prep["outlier"] == OUTLIER_MODES[0] else f" (k={prep['k']})")
    )
    c1, c2 = st.columns(2)
    test_size = c1.slider("시험용 데이터 비율", 0.1, 0.5, 0.2, 0.05)
    n_trees = c2.slider("랜덤 포레스트 나무 개수", 10, 300, 100, 10)

    model, X_te, y_te, pred = train_model(
        test_size, n_trees, prep["drop_dup"], prep["outlier"], prep["k"]
    )
    st.metric("정확도", f"{accuracy_score(y_te, pred) * 100:.2f}%")

    labels = [1, 2, 3, 4]
    names = [CLASS_KO[c] for c in labels]
    cm = confusion_matrix(y_te, pred, labels=labels)
    left, right = st.columns(2)
    with left:
        fig = px.imshow(
            cm, x=names, y=names, text_auto=True, color_continuous_scale="Blues",
            labels=dict(x="예측", y="실제", color="개수"), title="혼동 행렬",
        )
        fig.update_layout(height=420)
        st.plotly_chart(fig, width="stretch")
    with right:
        imp = pd.DataFrame({"센서": SENSORS, "중요도": model.feature_importances_})
        imp["각도"] = imp["센서"].map(ANGLES)
        imp = imp.sort_values("중요도")
        fig = px.bar(imp, x="중요도", y="센서", orientation="h",
                     hover_data=["각도"], title="센서 중요도")
        fig.update_layout(height=420)
        st.plotly_chart(fig, width="stretch")

    st.subheader("직접 넣어 보기")
    sample_idx = st.selectbox(
        "시험 데이터에서 한 줄 고르기 (값이 아래 칸에 채워져요)",
        X_te.index.tolist(),
        format_func=lambda i: f"id {df.loc[i, 'id']} · 실제 {CLASS_KO[int(df.loc[i, 'Class'])]}",
    )
    base = X_te.loc[sample_idx]

    inputs = {}
    grid = st.columns(6)
    for n, s in enumerate(SENSORS):
        inputs[s] = grid[n % 6].number_input(
            f"{s} ({ANGLES[s]}°)", value=float(base[s]), step=0.01,
            format="%.3f", key=f"in_{s}_{sample_idx}",
        )

    x_new = pd.DataFrame([inputs])[SENSORS]
    p = int(model.predict(x_new)[0])
    proba = model.predict_proba(x_new)[0]

    a, b = st.columns(2)
    with a:
        st.success(f"예측된 움직임: **{class_label(p)}**")
        pr = pd.DataFrame({"움직임": [CLASS_KO[c] for c in model.classes_], "확률": proba})
        fig = px.bar(pr, x="움직임", y="확률", text_auto=".2f", range_y=[0, 1])
        fig.update_layout(height=360)
        st.plotly_chart(fig, width="stretch")
    with b:
        st.plotly_chart(polar_chart(x_new.iloc[0], "센서 거리 (위쪽 = 앞 180°)"),
                        width="stretch")

# 5) 미로 시뮬레이션 --------------------------------------------------
else:
    st.header("🧩 미로 속 로봇 시뮬레이션")
    st.markdown(
        "무작위 미로에서 로봇이 **왼쪽 위(출발) → 오른쪽 아래(도착)** 로 갑니다. "
        "각 위치에서 24개 초음파 값을 계산해 **예측 df**에 넣고, 앞서 불러온 Robo.csv로 "
        "학습한 모델이 다음 움직임(Class 1~4)을 예측합니다. 로봇은 벽을 뚫고 지나가지 못합니다."
    )

    if "maze_seed" not in st.session_state:
        st.session_state.maze_seed = 2026

    c1, c2, c3 = st.columns(3)
    maze_rows = c1.slider("미로 세로 칸 수", 3, 12, 6)
    maze_cols = c2.slider("미로 가로 칸 수", 3, 12, 6)
    with c3:
        st.number_input("미로 번호(시드)", 0, 99999, key="maze_seed")
        st.button("🎲 새 미로 만들기", on_click=new_maze_seed)

    c4, c5, c6 = st.columns(3)
    unit_m = c4.slider("한 칸의 실제 크기(m)", 0.5, 3.0, 1.5, 0.1,
                       help="초음파 값(미터)을 데이터와 비슷한 범위로 맞추는 배율")
    orient_name = c5.radio("센서 번호 방향", ["시계 방향", "반시계 방향"], index=1, horizontal=True,
                           help="원본 데이터에서 V19 쪽 벽에 가까워지면 '약간 좌회전'을 하므로 "
                                "V19가 오른쪽 → 번호는 반시계 방향으로 늘어나는 것이 맞아요.")
    max_steps = c6.slider("최대 이동 칸 수", 50, 800, 400, 25)
    orient = 1 if orient_name == "시계 방향" else -1

    sensor_vals = df[SENSORS].astype(float)
    vmin, vmax = float(sensor_vals.min().min()), float(sensor_vals.max().max())
    st.caption(f"초음파 값 범위: 원본 데이터 기준 {vmin:.3f} ~ {vmax:.3f} m 로 자름")

    prep = get_prep()
    prep_key = (prep["drop_dup"], prep["outlier"], prep["k"])
    st.caption(
        f"전처리: 중복 제거 {'O' if prep['drop_dup'] else 'X'} · 이상치 {prep['outlier']}"
        + ("" if prep["outlier"] == OUTLIER_MODES[0] else f" (k={prep['k']})")
    )

    grid = make_maze(maze_rows, maze_cols, int(st.session_state.maze_seed))

    tab1, tab2 = st.tabs(["① 출발→도착 이동과 예측 df", "② 예측한 움직임대로 이동"])

    with tab1:
        path_df = simulate_path(grid, unit_m, vmin, vmax, orient, prep_key)
        st.plotly_chart(
            maze_animation(grid, path_df, unit_m, orient, "최단 경로를 따라 이동 (색 = 예측한 움직임)",
                           "예측 Class"),
            width="stretch",
        )
        st.subheader("예측 df")
        st.caption("경로 위 모든 위치(지금 위치 + 앞으로 지날 위치)의 초음파 예상값과 다음 움직임 예측")
        st.dataframe(path_df, hide_index=True, width="stretch", height=320)

        cnt = path_df["예측 Class"].value_counts().reindex([1, 2, 3, 4], fill_value=0).reset_index()
        cnt.columns = ["Class", "횟수"]
        cnt["움직임"] = cnt["Class"].map(lambda c: class_label(int(c)))
        fig = px.bar(cnt, x="움직임", y="횟수", color="움직임", text="횟수",
                     color_discrete_sequence=[CLASS_COLORS[c] for c in (1, 2, 3, 4)],
                     title="경로 전체에서 예측된 움직임 횟수")
        fig.update_layout(showlegend=False, height=340)
        st.plotly_chart(fig, width="stretch")

    with tab2:
        drive_df, reached, moves = simulate_model_drive(grid, unit_m, vmin, vmax, orient, max_steps, prep_key)
        st.caption("갈림길마다 모델 확률을 방향 점수로 써요 — 직진 = P(1), 오른쪽 = P(2)+P(3), 왼쪽 = P(4). "
                   "벽이 있는 칸으로는 가지 않고, 지나온 길은 피하며, 막다른 길이면 되돌아가요.")
        if reached:
            st.success(f"🎉 예측한 움직임대로 {moves}칸 움직여 도착했어요!")
        else:
            st.warning(f"{max_steps}칸 안에 도착하지 못했어요. '최대 이동 칸 수'를 늘려 보세요.")

        decided = drive_df[drive_df["비고"] != "출발"]
        follow = (decided["비고"] == "").mean() * 100 if len(decided) else 0.0
        m1, m2, m3 = st.columns(3)
        m1.metric("이동한 칸 수", moves)
        m2.metric("1순위 예측대로 간 비율", f"{follow:.1f}%")
        m3.metric("되돌아간 장면 수", int((drive_df["비고"] == "막다른 길 → 되돌아감").sum()))

        st.plotly_chart(
            maze_animation(grid, drive_df, unit_m, orient, "모델이 예측한 움직임대로 이동 (색 = 실행한 움직임)",
                           "실행 Class"),
            width="stretch",
        )
        st.subheader("이동 기록 df")
        st.dataframe(drive_df, hide_index=True, width="stretch", height=320)
