import pandas as pd

# 1. 파일 불러오기 (이 부분에서 에러가 난다면 파일이 같은 폴더/위치에 있는지 꼭 확인!)
try:
    df = pd.read_csv('../data/Student_data.csv')
    print("✅ 데이터 불러오기 성공!")
except FileNotFoundError:
    print("❌ 에러: 'Student_data.csv' 파일을 찾을 수 없습니다. 파일이 같은 폴더에 있는지 확인해주세요!")

# 데이터가 정상적으로 불러와졌을 때만 아래 분석 코드 실행
if 'df' in locals():
    # 2. 고학점 기준 정하기 (상위 20% 커트라인 점수 찾기)
    top_20_cutoff = df['Final_CGPA'].quantile(0.80)
    print(f"\n🎯 상위 20% 학점 커트라인: {top_20_cutoff:.2f}점")

    # 3. 고학점 그룹 vs 나머지 그룹 나누기
    top_students = df[df['Final_CGPA'] >= top_20_cutoff]
    normal_students = df[df['Final_CGPA'] < top_20_cutoff]

    # 4. 분석할 핵심 특징(변수) 리스트
    features = ['Attendance_Pct', 'Study_Hours_Per_Day', 'Sleep_Hours', 'Social_Hours_Week']

    # 5. 두 그룹의 평균 비교 (소수점 둘째 자리까지 깔끔하게 출력)
    print("\n=== 🏆 상위 20% 학생들의 생활 패턴 평균 ===")
    print(top_students[features].mean().round(2))

    print("\n=== 🏃 나머지 80% 학생들의 생활 패턴 평균 ===")
    print(normal_students[features].mean().round(2))