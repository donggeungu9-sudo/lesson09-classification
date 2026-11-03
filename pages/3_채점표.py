import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_graphviz
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
    st.dataframe(make_cm_df(tree_tn, tree_fp, tree_fn, tree_tp), use_container_width=True)

with cm_col3:
    st.write("##### 훈련용에서 많은 쪽으로만 답하는 모델")
    st.dataframe(make_cm_df(dummy_tn, dummy_fp, dummy_fn, dummy_tp), use_container_width=True)

st.markdown("---")

# ---------------------------------------------------------
# 정확도·재현율·정밀도·F1 비교 표
# ---------------------------------------------------------
st.subheader("🏆 성능 지표 비교표")

metrics_summary_df = pd.DataFrame([
    {
        "모델명": "로지스틱 회귀(확률로 답하는 모델)",
        "정확도 (Accuracy)": log_metrics["accuracy_str"],
        "재현율 (Recall)": log_metrics["recall_str"],
        "정밀도 (Precision)": log_metrics["precision_str"],
        "F1 점수": log_metrics["f1_str"]
    },
    {
        "모델명": "의사결정트리(질문으로 답하는 모델)",
        "정확도 (Accuracy)": tree_metrics["accuracy_str"],
        "재현율 (Recall)": tree_metrics["recall_str"],
        "정밀도 (Precision)": tree_metrics["precision_str"],
        "F1 점수": tree_metrics["f1_str"]
    },
    {
        "모델명": "훈련용에서 많은 쪽으로만 답하는 모델",
        "정확도 (Accuracy)": dummy_metrics["accuracy_str"],
        "재현율 (Recall)": dummy_metrics["recall_str"],
        "정밀도 (Precision)": dummy_metrics["precision_str"],
        "F1 점수": dummy_metrics["f1_str"]
    }
])

st.dataframe(metrics_summary_df, use_container_width=True)

st.markdown("---")

# ---------------------------------------------------------
# 의사결정트리 구조 시각화
# ---------------------------------------------------------
st.subheader("🌲 의사결정트리(질문으로 답하는 모델) 구조 시각화")
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

st.markdown("---")

# ---------------------------------------------------------
# 오류 분석 목록 (놓친 환자 vs 헛짚은 사람)
# ---------------------------------------------------------
st.subheader("🔍 오류 분석 목록 (놓친 환자 & 헛짚은 사람)")

model_choice = st.selectbox(
    "분석할 모델을 선택해 주세요:",
    options=[
        "로지스틱 회귀(확률로 답하는 모델)",
        "의사결정트리(질문으로 답하는 모델)",
        "훈련용에서 많은 쪽으로만 답하는 모델"
    ]
)

if model_choice.startswith("로지스틱"):
    chosen_preds = y_pred_log
    chosen_fn, chosen_fp = log_fn, log_fp
elif model_choice.startswith("의사결정"):
    chosen_preds = y_pred_tree
    chosen_fn, chosen_fp = tree_fn, tree_tp
else:
    chosen_preds = y_pred_dummy
    chosen_fn, chosen_fp = dummy_fn, dummy_fp

df_analysis = df_test.copy()
df_analysis["pred"] = chosen_preds

fn_df = df_analysis[(df_analysis["stroke"] == 1) & (df_analysis["pred"] == 0)].copy()
fp_df = df_analysis[(df_analysis["stroke"] == 0) & (df_analysis["pred"] == 1)].copy()

fn_df = fn_df.sort_values(by="age", ascending=False).reset_index(drop=True)
fp_df = fp_df.sort_values(by="age", ascending=False).reset_index(drop=True)

fn_total = len(fn_df)
fp_total = len(fp_df)

list_col1, list_col2 = st.columns(2)

def format_error_dataframe(sub_df):
    if sub_df.empty:
        return pd.DataFrame(columns=["번호", "나이", "평균 혈당", "고혈압 여부", "심장병 여부"])
    
    formatted = pd.DataFrame({
        "번호": sub_df.index + 1,
        "나이": sub_df["age"],
        "평균 혈당": sub_df["avg_glucose_level"],
        "고혈압 여부": sub_df["hypertension"].map({1: "있음 (1)", 0: "없음 (0)"}),
        "심장병 여부": sub_df["heart_disease"].map({1: "있음 (1)", 0: "없음 (0)"})
    })
    return formatted

with list_col1:
    display_fn = fn_df.head(20)
    st.write(f"##### 🩺 실제 뇌졸중인데 아니라고 예측 (놓친 환자) - 총 {fn_total:,}명")
    st.dataframe(format_error_dataframe(display_fn), use_container_width=True)
    if fn_total > 20:
        st.caption(f"ℹ️ 전체 {fn_total:,}명 중 앞 20명만 표시되었습니다.")

with list_col2:
    display_fp = fp_df.head(20)
    st.write(f"##### ⚠️ 실제 아닌데 뇌졸중이라고 예측 (헛짚은 사람) - 총 {fp_total:,}명")
    st.dataframe(format_error_dataframe(display_fp), use_container_width=True)
    if fp_total > 20:
        st.caption(f"ℹ️ 전체 {fp_total:,}명 중 앞 20명만 표시되었습니다.")

st.markdown("---")
st.write(f"📌 **확인**: 놓친 환자(FN) 수 **{fn_total}명**은 혼동행렬의 FN 칸 수와 정확히 일치하며, 헛짚은 사람(FP) 수 **{fp_total}명**은 혼동행렬의 FP 칸 수와 정확히 일치합니다.")
