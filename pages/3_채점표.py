import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier

# 페이지 설정
st.set_page_config(
    page_title="뇌졸중 예측 실습실 - 채점표",
    page_icon="📋",
    layout="wide"
)

st.title("📋 분류 모델 채점표 및 오류 분석")
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

if len(selected_korean) < 2:
    st.warning("⚠️ 속성을 최소 2개 이상 선택해야 모델 평가를 진행할 수 있습니다.")
    st.stop()

selected_cols = [FEATURE_MAP[k] for k in selected_korean]

# 데이터 분할 (10명 중 앞 3명 테스트용 고정 -> 1,533명 테스트셋)
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

# 예측 수행
y_pred_log = log_reg.predict(X_test_scaled)
y_pred_tree = tree_clf.predict(X_test_raw)
y_pred_dummy = dummy_clf.predict(X_test_raw)

# 테스트용 사람 수 및 실제 뇌졸중인 사람 수
test_total_count = len(y_test)
test_stroke_count = int(y_test.sum())

st.info(f"📊 **테스트용 데이터 인원수**: 총 **{test_total_count:,}명** | 그중 실제 뇌졸중 환자 수: **{test_stroke_count:,}명**")

# 혼동행렬 계산 함수 (TN, FP, FN, TP)
def get_confusion_matrix_counts(y_true, y_pred):
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    return tn, fp, fn, tp

log_tn, log_fp, log_fn, log_tp = get_confusion_matrix_counts(y_test, y_pred_log)
tree_tn, tree_fp, tree_fn, tree_tp = get_confusion_matrix_counts(y_test, y_pred_tree)
dummy_tn, dummy_fp, dummy_fn, dummy_tp = get_confusion_matrix_counts(y_test, y_pred_dummy)

# 지표 계산 함수
def compute_metrics(tn, fp, fn, tp):
    total = tn + fp + fn + tp
    accuracy = (tp + tn) / total if total > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    if (tp + fp) > 0:
        precision = tp / (tp + fp)
        if (precision + recall) > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0
        precision_str = f"{precision * 100:.2f}%"
        f1_str = f"{f1 * 100:.2f}%"
        is_calculable = True
    else:
        precision_str = "계산할 수 없음"
        f1_str = "계산할 수 없음"
        is_calculable = False
        precision = 0.0
        f1 = 0.0
        
    return {
        "accuracy": accuracy,
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "accuracy_str": f"{accuracy * 100:.2f}%",
        "recall_str": f"{recall * 100:.2f}%",
        "precision_str": precision_str,
        "f1_str": f1_str,
        "is_calculable": is_calculable
    }

log_metrics = compute_metrics(log_tn, log_fp, log_fn, log_tp)
tree_metrics = compute_metrics(tree_tn, tree_fp, tree_fn, tree_tp)
dummy_metrics = compute_metrics(dummy_tn, dummy_fp, dummy_fn, dummy_tp)

st.markdown("---")

# ---------------------------------------------------------
# 혼동행렬 표시 (2x2 표, 행: 실제, 열: 예측)
# ---------------------------------------------------------
st.subheader("📊 모델별 혼동행렬 (Confusion Matrix)")

cm_col1, cm_col2, cm_col3 = st.columns(3)

def make_cm_df(tn, fp, fn, tp):
    return pd.DataFrame([
        {
            "예측: 아님(0)": f"TN: {tn:,}명",
            "예측: 뇌졸중(1)": f"FP: {fp:,}명",
            "행 합계": f"{tn + fp:,}명"
        },
        {
            "예측: 아님(0)": f"FN: {fn:,}명",
            "예측: 뇌졸중(1)": f"TP: {tp:,}명",
            "행 합계": f"{fn + tp:,}명"
        }
    ], index=["실제: 아님(0)", "실제: 뇌졸중(1)"])

with cm_col1:
    st.write("##### 로지스틱 회귀(확률로 답하는 모델)")
    st.dataframe(make_cm_df(log_tn, log_fp, log_fn, log_tp), use_container_width=True)

with cm_col2:
    st.write("##### 의사결정트리(질문으로 답하는 모델)")
    st.dataframe
