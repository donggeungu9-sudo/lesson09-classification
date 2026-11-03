import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier

# 페이지 설정
st.set_page_config(
    page_title="뇌졸중 예측 실습실 - 분류 모델",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 분류 모델 평가 및 시각화")
st.markdown("---")

# 1. 데이터 불러오기
DATA_URL = "https://raw.githubusercontent.com/donggeungu9-sudo/data/refs/heads/main/stroke.csv"

@st.cache_data
def load_data():
    return pd.read_csv(DATA_URL, encoding="utf-8")

df_raw = load_data()

# 우리말 이름과 열 이름 매핑
FEATURE_MAP = {
    "나이": "age",
    "평균 혈당": "avg_glucose_level",
    "체질량지수": "bmi",
    "고혈압": "hypertension",
    "심장병": "heart_disease"
}
REVERSE_FEATURE_MAP = {v: k for k, v in FEATURE_MAP.items()}

# 속성 선택 multiselect (기본값: bmi 뺀 넷)
default_selected_korean = ["나이", "평균 혈당", "고혈압", "심장병"]

selected_korean = st.multiselect(
    "학습에 사용할 속성을 선택해 주세요 (2개 이상 선택):",
    options=list(FEATURE_MAP.keys()),
    default=default_selected_korean
)

# 2개 마 미만 고를 시 경고 및 중단
if len(selected_korean) < 2:
    st.warning("⚠️ 속성을 최소 2개 이상 선택해야 모델 학습 및 시각화를 진행할 수 있습니다.")
    st.stop()

selected_cols = [FEATURE_MAP[k] for k in selected_korean]

# 번호(index) 순으로 정렬 후 10명씩 묶어 앞 3명을 테스트용으로 고정 (1,533명 테스트셋)
df_sorted = df_raw.copy().reset_index(drop=True)
test_mask = (df_sorted.index % 10) < 3
train_mask = ~test_mask

df_train = df_sorted[train_mask].copy()
df_test = df_sorted[test_mask].copy()

# bmi 선택 시 결측치를 훈련용 데이터의 중앙값으로 채움
if "bmi" in selected_cols:
    bmi_median_train = df_train["bmi"].median()
    df_train["bmi"] = df_train["bmi"].fillna(bmi_median_train)
    df_test["bmi"] = df_test["bmi"].fillna(bmi_median_train)

X_train_raw = df_train[selected_cols]
y_train = df_train["stroke"]

X_test_raw = df_test[selected_cols]
y_test = df_test["stroke"]

# 특성 스케일링 (로지스틱 회귀용)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)
X_test_scaled = scaler.transform(X_test_raw)

# ---------------------------------------------------------
# 모델 학습
# ---------------------------------------------------------
log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_train_scaled, y_train)

tree_clf = DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, random_state=42)
tree_clf.fit(X_train_raw, y_train)

dummy_clf = DummyClassifier(strategy="most_frequent")
dummy_clf.fit(X_train_raw, y_train)

# 정확도 계산
log_train_acc = log_reg.score(X_train_scaled, y_train)
log_test_acc = log_reg.score(X_test_scaled, y_test)

tree_train_acc = tree_clf.score(X_train_raw, y_train)
tree_test_acc = tree_clf.score(X_test_raw, y_test)

dummy_train_acc = dummy_clf.score(X_train_raw, y_train)
dummy_test_acc = dummy_clf.score(X_test_raw, y_test)

# ---------------------------------------------------------
# 카드 3개로 모델 정확도 출력
# ---------------------------------------------------------
st.subheader("🎯 모델 정확도 비교")
c1, c2, c3 = st.columns(3)

with c1:
    st.metric(
        label="로지스틱 회귀(확률로 답하는 모델)",
        value=f"{log_test_acc * 100:.2f}%",
        delta=f"훈련: {log_train_acc * 100:.2f}% | 테스트: {log_test_acc * 100:.2f}%",
        delta_color="off"
    )

with c2:
    st.metric(
        label="의사결정트리(질문으로 답하는 모델)",
        value=f"{tree_test_acc * 100:.2f}%",
        delta=f"훈련: {tree_train_acc * 100:.2f}% | 테스트: {tree_test_acc * 100:.2f}%",
        delta_color="off"
    )

with c3:
    st.metric(
        label="입력을 하나도 보지 않고 훈련용에서 많은 쪽으로만 답하는 모델",
        value=f"{dummy_test_acc * 100:.2f}%",
        delta=f"훈련: {dummy_train_acc * 100:.2f}% | 테스트: {dummy_test_acc * 100:.2f}%",
        delta_color="off"
    )

st.markdown("---")

# ---------------------------------------------------------
# 결정 경계 및 산점도 시각화 (Plotly)
# ---------------------------------------------------------
st.subheader("🎨 결정 경계 시각화")

sc_col1, sc_col2 = st.columns(2)
with sc_col1:
    x_korean = st.selectbox("가로축(X) 속성 선택:", options=selected_korean, index=0)
with sc_col2:
    y_korean = st.selectbox("세로축(Y) 속성 선택:", options=selected_korean, index=min(1, len(selected_korean)-1))

if x_korean == y_korean:
    st.info("💡 가로축과 세로축 속성을 다르게 선택하면 더 직관적인 2차원 산점도를 확인할 수 있습니다.")

x_col = FEATURE_MAP[x_korean]
y_col = FEATURE_MAP[y_korean]

other_cols = [c for c in selected_cols if c not in [x_col, y_col]]
other_medians_test = {c: df_test[c].median() for c in other_cols}

if other_cols:
    fixed_text_list = [f"**{REVERSE_FEATURE_MAP[c]}**: {v:.2f}" for c, v in other_medians_test.items()]
    st.write("📌 **시각화 축 외 선택된 속성 고정값 (테스트 데이터 중앙값)**: " + ", ".join(fixed_text_list))
else:
    st.write("📌 선택된 2개의 속성을 그대로 가로축과 세로축에 매핑하였습니다.")

x_min, x_max = df_test[x_col].min(), df_test[x_col].max()
y_min, y_max = df_test[y_col].min(), df_test[y_col].max()

x_padding = (x_max - x_min) * 0.05 if x_max != x_min else 1.0
y_padding = (y_max - y_min) * 0.05 if y_max != y_min else 1.0

x_min_pad, x_max_pad = x_min - x_padding, x_max + x_padding
y_min_pad, y_max_pad = y_min - y_padding, y_max + y_padding

grid_res = 150
xx, yy = np.meshgrid(
    np.linspace(x_min_pad, x_max_pad, grid_res),
    np.linspace(y_min_pad, y_max_pad, grid_res)
)

grid_df = pd.DataFrame({
    x_col: xx.ravel(),
    y_col: yy.ravel()
})
for c in other_cols:
    grid_df[c] = other_medians_test[c]

grid_df = grid_df[selected_cols]

z_tree = tree_clf.predict(grid_df).reshape(xx.shape)

grid_scaled = scaler.transform(grid_df)
z_log_prob = log_reg.predict_proba(grid_scaled)[:, 1].reshape(xx.shape)

fig = go.Figure()

fig.add_trace(go.Contour(
    x=np.linspace(x_min_pad, x_max_pad, grid_res),
    y=np.linspace(y_min_pad, y_max_pad, grid_res),
    z=z_tree,
    showscale=False,
    opacity=0.25,
    colorscale=[[0, "#1f77b4"], [1, "#ff7f0e"]],
    hoverinfo="skip"
))

has_contour_line = (z_log_prob.min() <= 0.5 <= z_log_prob.max())

if has_contour_line:
    fig.add_trace(go.Contour(
        x=np.linspace(x_min_pad, x_max_pad, grid_res),
        y=np.linspace(y_min_pad, y_max_pad, grid_res),
        z=z_log_prob,
        contours=dict(start=0.5, end=0.5, coloring='lines'),
        line=dict(color='red', width=3, dash='dash'),
        name='로지스틱 회귀 결정 경계 (0.5)',
        showscale=False
    ))
else:
    st.info("ℹ️ 현재 선택된 속성 조합 및 고정값 조건에서는 로지스틱 회귀 모델의 0.5 확률 결정 경계선이 그림 범위를 벗어납니다.")

df_test_plot = df_test.copy()
normal_points = df_test_plot[df_test_plot["stroke"] == 0]
stroke_points = df_test_plot[df_test_plot["stroke"] == 1]

fig.add_trace(go.Scatter(
    x=normal_points[x_col],
    y=normal_points[y_col],
    mode='markers',
    name='아님 (0)',
    marker=dict(color='#1f77b4', size=7, opacity=0.8)
))

fig.add_trace(go.Scatter(
    x=stroke_points[x_col],
    y=stroke_points[y_col],
    mode='markers',
    name='뇌졸중 (1)',
    marker=dict(color='#ff7f0e', size=8, symbol='x', opacity=0.9)
))

fig.update_layout(
    title=f"테스트 데이터 산점도 및 결정 경계 ({x_korean} vs {y_korean})",
    xaxis_title=x_korean,
    yaxis_title=y_korean,
    legend_title="실제 뇌졸중 여부",
    height=600
)

st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# ---------------------------------------------------------
# 의사결정트리 구조 (export_graphviz를 활용한 표준 렌더링)
# ---------------------------------------------------------
st.subheader("🌲 의사결정트리 구조 시각화")

korean_feature_names = [REVERSE_FEATURE_MAP[col] for col in selected_cols]

dot_data = export_graphviz(
    tree_clf,
    out_file=None,
    feature_names=korean_feature_names,
    class_names=["아님", "뇌졸중"],
    filled=True,
    rounded=True,
    special_characters=True,
    precision=2
)

st.graphviz_chart(dot_data)

# 요약 정보 계산
tree_structure = tree_clf.tree_
leaf_nodes = [i for i in range(tree_structure.node_count) if tree_structure.children_left[i] == tree_structure.children_right[i]]
total_leaf_count = len(leaf_nodes)

no_stroke_leaf_count = 0
for leaf_id in leaf_nodes:
    val = tree_structure.value[leaf_id][0]
    if np.argmax(val) == 0:
        no_stroke_leaf_count += 1

used_feature_indices = set(tree_structure.feature[i] for i in range(tree_structure.node_count) if tree_structure.children_left[i] != tree_structure.children_right[i])
used_features_korean = [korean_feature_names[i] for i in used_feature_indices]

st.write(f"📌 **답을 내는 최종 마디(리프 노드)**: 모두 **{total_leaf_count}칸**이고 그중 **{no_stroke_leaf_count}칸**이 **아님**이라고 답합니다.")
st.write(f"📌 **고른 속성 가운데 이 나무가 실제로 물은 속성**: {', '.join(used_features_korean) if used_features_korean else '없음'}")
