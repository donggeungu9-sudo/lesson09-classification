import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier

# 페이지 설정
st.set_page_config(
    page_title="뇌졸중 예측 실습실 - 성능 높이기",
    page_icon="🚀",
    layout="wide"
)

st.title("🚀 성능 높이기 (클래스 가중치 조정)")
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

# 속성 선택 multiselect (기본값: age, avg_glucose_level, hypertension, heart_disease 넷)
default_selected_korean = ["나이", "평균 혈당", "고혈압", "심장병"]

selected_korean = st.multiselect(
    "학습에 사용할 속성을 선택해 주세요 (2개 이상 선택):",
    options=list(FEATURE_MAP.keys()),
    default=default_selected_korean
)

if len(selected_korean) < 2:
    st.warning("⚠️ 속성을 최소 2개 이상 선택해야 모델 학습 및 시각화를 진행할 수 있습니다.")
    st.stop()

selected_cols = [FEATURE_MAP[k] for k in selected_korean]

# 데이터 분할 (10명 중 앞 3명 테스트용 고정)
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
# 모델 학습 (가중치 전 / 후)
# ---------------------------------------------------------
# 1) 로지스틱 회귀
log_reg_before = LogisticRegression(random_state=42)
log_reg_before.fit(X_train_scaled, y_train)

log_reg_after = LogisticRegression(class_weight="balanced", random_state=42)
log_reg_after.fit(X_train_scaled, y_train)

# 2) 의사결정트리
tree_before = DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, random_state=42)
tree_before.fit(X_train_raw, y_train)

tree_after = DecisionTreeClassifier(class_weight="balanced", max_depth=3, min_samples_leaf=5, random_state=42)
tree_after.fit(X_train_raw, y_train)

# 예측 수행
y_pred_log_b = log_reg_before.predict(X_test_scaled)
y_pred_log_a = log_reg_after.predict(X_test_scaled)

y_pred_tree_b = tree_before.predict(X_test_raw)
y_pred_tree_a = tree_after.predict(X_test_raw)

# 혼동행렬 지표 계산 헬퍼
def get_cm_and_metrics(y_true, y_pred):
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    
    total = tn + fp + fn + tp
    accuracy = (tp + tn) / total if total > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    if (tp + fp) > 0:
        precision = tp / (tp + fp)
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        precision_str = f"{precision * 100:.2f}%"
        f1_str = f"{f1 * 100:.2f}%"
    else:
        precision_str = "계산할 수 없음"
        f1_str = "계산할 수 없음"
        
    return {
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "accuracy_str": f"{accuracy * 100:.2f}%",
        "recall_str": f"{recall * 100:.2f}%",
        "precision_str": precision_str,
        "f1_str": f1_str
    }

log_b_m = get_cm_and_metrics(y_test, y_pred_log_b)
log_a_m = get_cm_and_metrics(y_test, y_pred_log_a)

tree_b_m = get_cm_and_metrics(y_test, y_pred_tree_b)
tree_a_m = get_cm_and_metrics(y_test, y_pred_tree_a)

# ---------------------------------------------------------
# 화면 출력: 전·후 비교표 및 혼동행렬
# ---------------------------------------------------------
st.subheader("📊 클래스 가중치 적용 전·후 모델 성능 비교")

def render_comparison_section(model_name, b_m, b_preds, a_m, a_preds):
    st.write(f"#### 🤖 {model_name}")
    col_b, col_a = st.columns(2)
    
    with col_b:
        st.markdown("**[가중치 적용 전]**")
        cm_df_b = pd.DataFrame([
            {"예측: 아님(0)": f"TN: {b_m['tn']:,}", "예측: 뇌졸중(1)": f"FP: {b_m['fp']:,}"},
            {"예측: 아님(0)": f"FN: {b_m['fn']:,}", "예측: 뇌졸중(1)": f"TP: {b_m['tp']:,}"}
        ], index=["실제: 아님(0)", "실제: 뇌졸중(1)"])
        st.dataframe(cm_df_b, use_container_width=True)
        
        metrics_df_b = pd.DataFrame([{
            "정확도": b_m['accuracy_str'],
            "재현율": b_m['recall_str'],
            "정밀도": b_m['precision_str'],
            "F1점수": b_m['f1_str']
        }])
        st.dataframe(metrics_df_b, use_container_width=True)
        
    with col_a:
        st.markdown("**[가중치 적용 후 (balanced)]**")
        cm_df_a = pd.DataFrame([
            {"예측: 아님(0)": f"TN: {a_m['tn']:,}", "예측: 뇌졸중(1)": f"FP: {a_m['fp']:,}"},
            {"예측: 아님(0)": f"FN: {a_m['fn']:,}", "예측: 뇌졸중(1)": f"TP: {a_m['tp']:,}"}
        ], index=["실제: 아님(0)", "실제: 뇌졸중(1)"])
        st.dataframe(cm_df_a, use_container_width=True)
        
        metrics_df_a = pd.DataFrame([{
            "정확도": a_m['accuracy_str'],
            "재현율": a_m['recall_str'],
            "정밀도": a_m['precision_str'],
            "F1점수": a_m['f1_str']
        }])
        st.dataframe(metrics_df_a, use_container_width=True)
        
    st.info(f"💡 **변화 요약**: 찾은 환자 수(TP)는 **{b_m['tp']}명**에서 **{a_m['tp']}명**으로, 놓친 환자 수(FN)는 **{b_m['fn']}명**에서 **{a_m['fn']}명**으로, 헛짚은 사람 수(FP)는 **{b_m['fp']}명**에서 **{a_m['fp']}명**으로 바뀌었습니다.")
    st.markdown("---")

render_comparison_section("로지스틱 회귀(확률로 답하는 모델)", log_b_m, y_pred_log_b, log_a_m, y_pred_log_a)
render_comparison_section("의사결정트리(질문으로 답하는 모델)", tree_b_m, y_pred_tree_b, tree_a_m, y_pred_tree_a)

# ---------------------------------------------------------
# 산점도 및 결정 경계 / 평면 시각화
# ---------------------------------------------------------
st.subheader("🎨 로지스틱 회귀 가중치 전·후 결정 경계 시각화")

if len(selected_korean) == 2:
    sc1, sc2 = st.columns(2)
    with sc1:
        x_korean = st.selectbox("가로축(X) 속성:", options=selected_korean, index=0)
    with sc2:
        y_korean = st.selectbox("세로축(Y) 속성:", options=selected_korean, index=1)
        
    x_col = FEATURE_MAP[x_korean]
    y_col = FEATURE_MAP[y_korean]
    
    other_cols = [c for c in selected_cols if c not in [x_col, y_col]]
    other_medians = {c: df_test[c].median() for c in other_cols}
    
    x_min, x_max = df_test[x_col].min(), df_test[x_col].max()
    y_min, y_max = df_test[y_col].min(), df_test[y_col].max()
    
    x_pad = (x_max - x_min) * 0.05 if x_max != x_min else 1.0
    y_pad = (y_max - y_min) * 0.05 if y_max != y_min else 1.0
    
    grid_res = 150
    xx, yy = np.meshgrid(
        np.linspace(x_min - x_pad, x_max + x_pad, grid_res),
        np.linspace(y_min - y_pad, y_max + y_pad, grid_res)
    )
    
    grid_df = pd.DataFrame({x_col: xx.ravel(), y_col: yy.ravel()})
    for c in other_cols:
        grid_df[c] = other_medians[c]
    grid_df = grid_df[selected_cols]
    
    grid_scaled = scaler.transform(grid_df)
    z_b = log_reg_before.predict_proba(grid_scaled)[:, 1].reshape(xx.shape)
    z_a = log_reg_after.predict_proba(grid_scaled)[:, 1].reshape(xx.shape)
    
    fig2d = go.Figure()
    
    if z_b.min() <= 0.5 <= z_b.max():
        fig2d.add_trace(go.Contour(
            x=np.linspace(x_min - x_pad, x_max + x_pad, grid_res),
            y=np.linspace(y_min - y_pad, y_max + y_pad, grid_res),
            z=z_b,
            contours=dict(start=0.5, end=0.5, coloring='lines'),
            line=dict(color='blue', width=3, dash='dash'),
            name='적용 전 0.5 경계선',
            showscale=False
        ))
    else:
        st.info("ℹ️ 가중치 적용 전 로지스틱 회귀의 0.5 결정 경계선이 현재 2차원 그림 범위를 벗어납니다.")
        
    if z_a.min() <= 0.5 <= z_a.max():
        fig2d.add_trace(go.Contour(
            x=np.linspace(x_min - x_pad, x_max + x_pad, grid_res),
            y=np.linspace(y_min - y_pad, y_max + y_pad, grid_res),
            z=z_a,
            contours=dict(start=0.5, end=0.5, coloring='lines'),
            line=dict(color='red', width=3, dash='solid'),
            name='적용 후(balanced) 0.5 경계선',
            showscale=False
        ))
    else:
        st.info("ℹ️ 가중치 적용 후 로지스틱 회귀의 0.5 결정 경계선이 현재 2차원 그림 범위를 벗어납니다.")
        
    norm_pts = df_test[df_test["stroke"] == 0]
    stroke_pts = df_test[df_test["stroke"] == 1]
    
    fig2d.add_trace(go.Scatter(
        x=norm_pts[x_col], y=norm_pts[y_col], mode='markers',
        name='아님 (0)', marker=dict(color='#1f77b4', size=7, opacity=0.7)
    ))
    fig2d.add_trace(go.Scatter(
        x=stroke_pts[x_col], y=stroke_pts[y_col], mode='markers',
        name='뇌졸중 (1)', marker=dict(color='#ff7f0e', size=8, symbol='x', opacity=0.9)
    ))
    
    fig2d.update_layout(
        title=f"로지스틱 회귀 가중치 전·후 0.5 결정 경계 비교 ({x_korean} vs {y_korean})",
        xaxis_title=x_korean, yaxis_title=y_korean, height=600
    )
    st.plotly_chart(fig2d, use_container_width=True)

else:
    st.write("##### 🧊 3차원 산점도 시각화 (속성 3개 이상 선택 시)")
    axis_opts = selected_korean[:3]
    x_k, y_k, z_k = axis_opts[0], axis_opts[1], axis_opts[2]
    x_c, y_c, z_c = FEATURE_MAP[x_k], FEATURE_MAP[y_k], FEATURE_MAP[z_k]
    
    fig3d = go.Figure()
    norm_pts = df_test[df_test["stroke"] == 0]
    stroke_pts = df_test[df_test["stroke"] == 1]
    
    fig3d.add_trace(go.Scatter3d(
        x=norm_pts[x_c], y=norm_pts[y_c], z=norm_pts[z_c],
        mode='markers', name='아님 (0)', marker=dict(size=4, color='#1f77b4', opacity=0.6)
    ))
    fig3d.add_trace(go.Scatter3d(
        x=stroke_pts[x_c], y=stroke_pts[y_c], z=stroke_pts[z_c],
        mode='markers', name='뇌졸중 (1)', marker=dict(size=6, color='#ff7f0e', symbol='x', opacity=0.9)
    ))
    
    fig3d.update_layout(
        title=f"3차원 산점도 ({x_k}, {y_k}, {z_k})",
        scene=dict(xaxis_title=x_k, yaxis_title=y_k, zaxis_title=z_k),
        height=700
    )
    st.plotly_chart(fig3d, use_container_width=True)
    st.info("ℹ️ 3차원 보기에서는 마우스로 회전하며 데이터 분포와 환자 분포를 다각도로 확인할 수 있습니다.")
