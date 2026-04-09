import os
import pandas as pd

# 현재 파일 위치 찾기
base_dir = os.path.dirname(__file__)

# 데이터 파일 접근
file_path = os.path.join(base_dir, "..", "data", "NetFlix.csv")

# 판다스로 읽기
df = pd.read_csv(file_path)

# 테스트용
print(df.head())