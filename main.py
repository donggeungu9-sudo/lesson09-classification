import os
import json
import pandas as pd
import streamlit as st
import plotly.express as px

# 1. 페이지 설정 (탭 제목 및 아이콘)
st.set_page_config(
    page_title="뇌졸중 예측 실습실",
    page_icon="🧠",
    layout="wide"
)

# 헤더 영역
st.title("🧠 뇌졸중 예측 실습실")
st.markdown("---")

# 2. 데이터 불러오기 (캐싱 적용, 요청하신 주소로 변경)
DATA_URL = "https://raw.githubusercontent.com/donggeungu9-sudo/data/refs/heads/main/stroke.csv"

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8")
    return df

df = load_data()

# 3. JSON 저장 및 불러오기 설정
SAVE_DIR = "내_기록"
SAVE_FILE = os.path.join(SAVE_DIR, "data_info.json")

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

def load_user_notes():
    if os.path.exists(SAVE_FILE):
        try:
            with open(SAVE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"meanings": {}, "source": ""}

def save_user_notes(meanings_dict, source_text):
    data = {
        "meanings": meanings_dict,
        "source": source_text
    }
    with open(SAVE_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

user_notes = load_user_notes()

# 4. 데이터 요약 카드 (Metric Card 4개)
total_count = len(df)
col_count = len(df.columns)
stroke_count = int(df["stroke"].sum())
stroke_ratio = (stroke_count / total_count) * 100

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric(label="전체 사람 수", value=f"{total_count:,}명")
with c2:
    st.metric(label="열 개수", value=f"{col_count}개")
with c3:
    st.metric(label="stroke 1 (환자 수)", value=f"{stroke_count:,}명")
with c4:
    st.metric(label="뇌졸중 비율", value=f"{stroke_ratio:.2f}%")

st.markdown("###")

# 5. 열 이름·우리말 뜻·값의 종류·빈 값 개수를 담은 표
st.subheader("📋 데이터 열 메타데이터")

null_counts = df.isnull().sum()
dtypes = df.dtypes

meanings_saved = user_notes.get("meanings", {})

table_data = []
for col in df.columns:
    table_data.append({
        "열 이름": col,
        "우리말 뜻": meanings_saved.get(col, ""),
        "값의 종류": str(dtypes[col]),
        "빈 값 개수": int(null_counts[col])
    })

meta_df = pd.DataFrame(table_data)

edited_meta_df = st.data_editor(
    meta_df,
    num_rows="fixed",
    use_container_width=True,
    column_config={
        "열 이름": st.column_config.Column(disabled=True),
        "값의 종류": st.column_config.Column(disabled=True),
        "빈 값 개수": st.column_config.Column(disabled=True),
        "우리말 뜻": st.column_config.TextColumn(
            help="교재를 보고 직접 채워 넣으세요."
        )
    },
    key="meta_editor"
)

# 6. 상위 5개 데이터 출력
st.subheader("🔍 데이터 5줄 미리보기")
st.dataframe(df.head(5), use_container_width=True)

# 7. 데이터 출처 작성란
st.subheader("📚 데이터 출처")
source_input = st.text_area(
    "데이터 출처를 작성해 주세요 (교재 내용 참고):",
    value=user_notes.get("source", ""),
    height=100,
    key="source_input"
)

# 8. 저장 버튼 (내_기록/data_info.json)
if st.button("💾 입력 내용 저장하기", type="primary"):
    updated_meanings = {}
    for idx, row in edited_meta_df.iterrows():
        updated_meanings[row["열 이름"]] = str(row["우리말 뜻"]) if row["우리말 뜻"] is not None else ""
    
    save_user_notes(updated_meanings, source_input)
    st.success("내_기록/data_info.json 에 성공적으로 저장되었습니다!")
