import pandas as pd
import streamlit as st
import plotly.express as px

# 페이지 설정
st.set_page_config(
    page_title="뇌졸중 예측 실습실 - 데이터 탐색",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 데이터 탐색 페이지")
st.markdown("---")

# 1. 동일한 데이터 동일 URL 및 UTF-8 인코딩으로 불러오기 (요청하신 주소 적용)
DATA_URL = "https://raw.githubusercontent.com/donggeungu9-sudo/data/refs/heads/main/stroke.csv"

@st.cache_data
def load_data():
    return pd.read_csv(DATA_URL, encoding="utf-8")

df = load_data()

# 분석 편의를 위해 stroke를 범주형 라벨 표기로 추가 생성 (그래프용)
df['stroke_label'] = df['stroke'].map({1: '뇌졸중 (1)', 0: '정상 (0)'})

# ---------------------------------------------------------
# 1. 나이와 평균 혈당의 분포 (히스토그램 두 개 나란히)
# ---------------------------------------------------------
st.subheader("📊 1. 나이 및 평균 혈당 분포 (히스토그램)")
col1, col2 = st.columns(2)

with col1:
    fig_age_hist = px.histogram(
        df,
        x="age",
        nbins=30,
        title="나이(age) 분포",
        labels={"age": "나이", "count": "인원 수"},
        color_discrete_sequence=["#1f77b4"]
    )
    st.plotly_chart(fig_age_hist, use_container_width=True)

with col2:
    fig_glucose_hist = px.histogram(
        df,
        x="avg_glucose_level",
        nbins=30,
        title="평균 혈당(avg_glucose_level) 분포",
        labels={"avg_glucose_level": "평균 혈당", "count": "인원 수"},
        color_discrete_sequence=["#ff7f0e"]
    )
    st.plotly_chart(fig_glucose_hist, use_container_width=True)

st.markdown("---")

# ---------------------------------------------------------
# 2. 뇌졸중 여부에 따른 나이와 평균 혈당 비교 (상자그림 & 평균값 표)
# ---------------------------------------------------------
st.subheader("📦 2. 뇌졸중 여부별 나이 및 평균 혈당 비교 (상자그림 & 평균표)")

box_col1, box_col2 = st.columns(2)

with box_col1:
    fig_age_box = px.box(
        df,
        x="stroke_label",
        y="age",
        color="stroke_label",
        title="뇌졸중 여부별 나이 분포",
        labels={"stroke_label": "뇌졸중 여부", "age": "나이"}
    )
    st.plotly_chart(fig_age_box, use_container_width=True)

with box_col2:
    fig_glucose_box = px.box(
        df,
        x="stroke_label",
        y="avg_glucose_level",
        color="stroke_label",
        title="뇌졸중 여부별 평균 혈당 분포",
        labels={"stroke_label": "뇌졸중 여부", "avg_glucose_level": "평균 혈당"}
    )
    st.plotly_chart(fig_glucose_box, use_container_width=True)

# 두 그룹의 평균값 표
mean_summary = df.groupby("stroke_label")[["age", "avg_glucose_level"]].mean().reset_index()
mean_summary.columns = ["뇌졸중 여부", "나이 평균", "평균 혈당 평균"]
mean_summary["나이 평균"] = mean_summary["나이 평균"].round(2)
mean_summary["평균 혈당 평균"] = mean_summary["평균 혈당 평균"].round(2)

st.write("##### 📌 뇌졸중 여부별 평균값 비교")
st.dataframe(mean_summary, use_container_width=True)

st.markdown("---")

# ---------------------------------------------------------
# 3. 고혈압 및 심장병 유무에 따른 뇌졸중 비율 (막대그래프)
# ---------------------------------------------------------
st.subheader("📈 3. 고혈압 및 심장병 유무별 뇌졸중 비율 (막대그래프)")

bar_col1, bar_col2 = st.columns(2)

# 고혈압별 뇌졸중 비율
hyp_stroke = df.groupby("hypertension")["stroke"].mean().reset_index()
hyp_stroke["hypertension_label"] = hyp_stroke["hypertension"].map({1: "고혈압 있음 (1)", 0: "고혈압 없음 (0)"})
hyp_stroke["stroke_ratio_pct"] = (hyp_stroke["stroke"] * 100).round(2)

with bar_col1:
    fig_hyp_bar = px.bar(
        hyp_stroke,
        x="hypertension_label",
        y="stroke_ratio_pct",
        text="stroke_ratio_pct",
        title="고혈압 유무별 뇌졸중 비율 (%)",
        labels={"hypertension_label": "고혈압 여부", "stroke_ratio_pct": "뇌졸중 비율 (%)"},
        color="hypertension_label"
    )
    fig_hyp_bar.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
    st.plotly_chart(fig_hyp_bar, use_container_width=True)

# 심장병별 뇌졸중 비율
heart_stroke = df.groupby("heart_disease")["stroke"].mean().reset_index()
heart_stroke["heart_label"] = heart_stroke["heart_disease"].map({1: "심장병 있음 (1)", 0: "심장병 없음 (0)"})
heart_stroke["stroke_ratio_pct"] = (heart_stroke["stroke"] * 100).round(2)

with bar_col2:
    fig_heart_bar = px.bar(
        heart_stroke,
        x="heart_label",
        y="stroke_ratio_pct",
        text="stroke_ratio_pct",
        title="심장병 유무별 뇌졸중 비율 (%)",
        labels={"heart_label": "심장병 여부", "stroke_ratio_pct": "뇌졸중 비율 (%)"},
        color="heart_label"
    )
    fig_heart_bar.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
    st.plotly_chart(fig_heart_bar, use_container_width=True)

st.markdown("---")

# ---------------------------------------------------------
# 4. bmi 결측치 그룹 vs 전체 그룹의 뇌졸중 비율 비교 표
# ---------------------------------------------------------
st.subheader("🩺 4. BMI 결측 대상자 vs 전체 뇌졸중 비율 비교")

bmi_null_df = df[df["bmi"].isnull()]

bmi_null_count = len(bmi_null_df)
bmi_null_stroke_ratio = (bmi_null_df["stroke"].mean() * 100) if bmi_null_count > 0 else 0

total_count = len(df)
total_stroke_ratio = (df["stroke"].mean() * 100)

bmi_comparison_df = pd.DataFrame([
    {
        "구분": "BMI 비어 있음 (결측치)",
        "사람 수 (명)": bmi_null_count,
        "뇌졸중 환자 수 (명)": int(bmi_null_df["stroke"].sum()),
        "뇌졸중 비율 (%)": round(bmi_null_stroke_ratio, 2)
    },
    {
        "구분": "전체 데이터",
        "사람 수 (명)": total_count,
        "뇌졸중 환자 수 (명)": int(df["stroke"].sum()),
        "뇌졸중 비율 (%)": round(total_stroke_ratio, 2)
    }
])

st.dataframe(bmi_comparison_df, use_container_width=True)

st.markdown("---")

# ---------------------------------------------------------
# 5. 흡연 상태(smoking_status) 값별 사람 수 표
# ---------------------------------------------------------
st.subheader("🚬 5. 흡연 상태(smoking_status)별 인원 수")

smoking_counts = df["smoking_status"].value_counts(dropna=False).reset_index()
smoking_counts.columns = ["흡연 상태 (smoking_status)", "사람 수 (명)"]

st.dataframe(smoking_counts, use_container_width=True)
