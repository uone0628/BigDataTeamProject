import os
import pandas as pd

# pd.set_option('display.max_columns', None)

# 현재 파일 위치 찾기
base_dir = os.path.dirname(__file__)

# 데이터 파일 접근
file_path = os.path.join(base_dir, "..", "data", "NetFlix.csv")

# 판다스로 읽기
df = pd.read_csv(file_path)

# 테스트용
print(df.head())

# 컬럼 이름 목록
print(df.columns.tolist())

# 분석에 사용할 컬럼 이름 list
needed_columns = ['type', 'title', 'cast', 'rating', 'genres']

# 새로운 데이터프레임 생성 (컬럼이 한 개 이상이라 대괄호 두 개가 맞음)
df_clean = df[needed_columns]

# 5개 다시 확인
print("필요한 컬럼만 뽑은 결과:")
print(df_clean.head())