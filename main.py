import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

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


# ─────────────────────────── 모델 ───────────────────────────
@st.cache_resource(show_spinner="모델을 학습하는 중...")
def train_model(test_size: float, n_trees: int):
    df = load_data().dropna(subset=["Class"])
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


# ─────────────────────────── 화면 ───────────────────────────
st.title(f"{APP_ICON} {APP_TITLE}")

try:
    df = load_data()
except Exception as e:  # noqa: BLE001
    st.error(f"데이터를 불러오지 못했어요: {e}")
    st.stop()

page = st.sidebar.radio("화면", ["데이터 소개", "상관관계", "움직임 분류"])
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
    st.dataframe(df.head(5), hide_index=True, use_container_width=True)

    st.subheader("움직임별 데이터 개수")
    counts = df["Class"].value_counts().sort_index().reset_index()
    counts.columns = ["Class", "개수"]
    counts["움직임"] = counts["Class"].map(lambda c: class_label(int(c)))
    fig = px.bar(counts, x="움직임", y="개수", color="움직임", text="개수")
    fig.update_layout(showlegend=False, height=380)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("열 설명 (우리말 뜻 · 출처)")
    st.caption("칸을 고치면 `내_기록/열_설명.json` 에 바로 저장되고, 새로고침해도 그대로 남아요.")

    if "records" not in st.session_state:
        st.session_state.records = load_records()

    edited = st.data_editor(
        st.session_state.records,
        key="record_editor",
        hide_index=True,
        disabled=["열 이름"],
        use_container_width=True,
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
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("숫자 표로 보기"):
        st.dataframe(corr.round(3), use_container_width=True)

    st.subheader("Class와의 상관계수")
    cc = corr["Class"].drop("Class").reset_index()
    cc.columns = ["열", "상관계수"]
    cc["설명"] = cc["열"].map(lambda c: f"{ANGLES[c]}°" if c in ANGLES else c)
    fig2 = px.bar(
        cc, x="열", y="상관계수", color="상관계수", hover_data=["설명"],
        color_continuous_scale="RdBu_r", range_color=[-1, 1],
    )
    fig2.update_layout(height=380)
    st.plotly_chart(fig2, use_container_width=True)

# 3) 움직임 분류 ------------------------------------------------------
else:
    st.header("센서값으로 움직임 맞히기")
    c1, c2 = st.columns(2)
    test_size = c1.slider("시험용 데이터 비율", 0.1, 0.5, 0.2, 0.05)
    n_trees = c2.slider("랜덤 포레스트 나무 개수", 10, 300, 100, 10)

    model, X_te, y_te, pred = train_model(test_size, n_trees)
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
        st.plotly_chart(fig, use_container_width=True)
    with right:
        imp = pd.DataFrame({"센서": SENSORS, "중요도": model.feature_importances_})
        imp["각도"] = imp["센서"].map(ANGLES)
        imp = imp.sort_values("중요도")
        fig = px.bar(imp, x="중요도", y="센서", orientation="h",
                     hover_data=["각도"], title="센서 중요도")
        fig.update_layout(height=420)
        st.plotly_chart(fig, use_container_width=True)

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
        st.plotly_chart(fig, use_container_width=True)
    with b:
        st.plotly_chart(polar_chart(x_new.iloc[0], "센서 거리 (위쪽 = 앞 180°)"),
                        use_container_width=True)
