import os
import pandas as pd

# 1. 'code' 폴더의 위치를 기준으로 'data' 폴더 안의 파일 경로를 정확히 찾아냅니다.
base_dir = os.path.dirname(__file__)
file_path = os.path.join(base_dir, "..", "data", "Student_data.csv")

# 2. 파일 불러오기
try:
    df = pd.read_csv(file_path) # 수정된 부분: file_path 변수를 넣습니다.
    print("✅ 데이터 불러오기 성공!")
except FileNotFoundError:
    print("❌ 에러: 'Student_data.csv' 파일을 찾을 수 없습니다.")

# 데이터가 정상적으로 불러와졌을 때만 아래 분석 코드 실행
if 'df' in locals():
    # 3. 고학점 기준 정하기 (상위 20% 커트라인 점수 찾기)
    top_20_cutoff = df['Final_CGPA'].quantile(0.80)
    print(f"\n🎯 상위 20% 학점 커트라인: {top_20_cutoff:.2f}점")

    # 4. 고학점 그룹 vs 나머지 그룹 나누기
    top_students = df[df['Final_CGPA'] >= top_20_cutoff]
    normal_students = df[df['Final_CGPA'] < top_20_cutoff]

    # 5. 분석할 핵심 특징(변수) 리스트
    features = ['Attendance_Pct', 'Study_Hours_Per_Day', 'Sleep_Hours', 'Social_Hours_Week']

    # 6. 두 그룹의 평균 비교 (소수점 둘째 자리까지 깔끔하게 출력)
    print("\n=== 🏆 상위 20% 학생들의 생활 패턴 평균 ===")
    print(top_students[features].mean().round(2))

    print("\n=== 🏃 나머지 80% 학생들의 생활 패턴 평균 ===")
    print(normal_students[features].mean().round(2))