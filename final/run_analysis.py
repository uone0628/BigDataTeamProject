import sqlite3
import os
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 기본 깨짐 방지 설정
plt.rcParams['font.family'] = 'AppleGothic'  
plt.rcParams['axes.unicode_minus'] = False

print("--- 성적 변화 수치 추론통계 및 시각화 ---")

# SQLite DB 연결
conn = sqlite3.connect('student_project.db')

# 캐글 데이터랑 설문 데이터 불러옴
df_kaggle = pd.read_csv('data/Student_data_processed.csv')
df_school_raw = pd.read_csv('data/Handong_data.csv')

# 한동 데이터 전처리 (공부시간 문자열 -> 숫자 변환) - .5로 바꿔서 계산
study_mapping = {
    '1시간 미만': 0.5, '1~2시간': 1.5, '2~3시간': 2.5, '3~4시간': 3.5,
    '4~5시간': 4.5, '5~6시간': 5.5, '6~7시간': 6.5, '7시간 이상': 7.5
}
df_school_raw['Study'] = df_school_raw['하루 평균 공부시간(순공시간)'].map(study_mapping)

# 직전 학기 성적과 그 전학기 성적 컬럼
df_school_processed = df_school_raw[['성별 ', '전공이 무엇인가요?', 'Study', '직전 학기 성적 (ex. 25-2)', '그 전학기 성적 (ex. 25-1)']].copy()
df_school_processed.columns = ['Gender', 'Major', 'Study', 'F_CGPA', 'P_CGPA']

# 숫자로 변환 (에러 방지) 및 결측치 제거
df_school_processed['F_CGPA'] = pd.to_numeric(df_school_processed['F_CGPA'], errors='coerce')
df_school_processed['P_CGPA'] = pd.to_numeric(df_school_processed['P_CGPA'], errors='coerce')
df_school_processed = df_school_processed.dropna(subset=['Study', 'F_CGPA', 'P_CGPA'])

# 아직 성적 기록이 없는 신입생/예외값(0점) 제외
df_school_processed = df_school_processed[(df_school_processed['F_CGPA'] > 0) & (df_school_processed['P_CGPA'] > 0)]

# 성적 변화량 계산 공식 적용 (직전 학기 성적 - 그 전학기 성적)
df_school_processed['CGPA_Diff'] = df_school_processed['F_CGPA'] - df_school_processed['P_CGPA']

# SQLite 테이블 적재
df_kaggle.to_sql('KAGGLE_DATA', conn, if_exists='replace', index=False)
df_school_processed.to_sql('SCHOOL_DATA', conn, if_exists='replace', index=False)
print("SQLite DB 적재 완료됨")


# 시각화 1 - 캐글 5,000명 성적 변화량(CGPA_Diff)의 확률분포
print("\n시각화 1 - 글로벌 데이터 성적 변화량 확률분포 시각화")

kaggle_diff_array = pd.read_sql_query("SELECT CGPA_Diff FROM KAGGLE_DATA", conn)['CGPA_Diff'].to_numpy()
kg_diff_mean = np.mean(kaggle_diff_array)
kg_diff_std = np.std(kaggle_diff_array, ddof=1)

plt.figure(figsize=(9, 4.5))
sns.histplot(kaggle_diff_array, kde=True, color='lightgreen', stat='density', alpha=0.6, label='실제 대규모 변화량 분포')

# 정규분포 수식 구현
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = (1 / (kg_diff_std * math.sqrt(2 * math.pi))) * np.exp(-0.5 * ((x - kg_diff_mean) / kg_diff_std) ** 2)
plt.plot(x, p, 'r--', linewidth=2, label='이론적 정규분포 곡선')

plt.title('캐글 데이터 성적 변화량(CGPA Diff)의 확률분포 (N=5,000)', fontsize=13, fontweight='bold')
plt.xlabel('성적 변화량 (직전학기 - 그 전학기)')
plt.ylabel('밀도 (Density)')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.5)
plt.tight_layout()
plt.savefig('kaggle_grade_diff_distribution.png', dpi=300)
plt.close()


# 시각화 2 - 공부 시간에 따른 한동대 성적 변화 수치
print("\n시각화 2 - 공부시간대별 실제 성적 상승/하락 수치 시각화")

# SQL GROUP BY를 통해 공부시간별 '평균 성적 변화량(CGPA_Diff)' 추출
query_group = """
SELECT Study, AVG(CGPA_Diff) AS Avg_Grade_Change
FROM SCHOOL_DATA 
GROUP BY Study 
ORDER BY Study
"""
summary_df = pd.read_sql_query(query_group, conn)

study_levels = summary_df['Study'].to_numpy()
avg_changes = summary_df['Avg_Grade_Change'].to_numpy()

fig, ax = plt.subplots(figsize=(9, 5))

# 바 플롯으로 성적 변화량 표현 - 0보다 크면 성적 상승, 작으면 하락한 것
colors = ['red' if val < 0 else 'royalblue' for val in avg_changes]
bars = ax.bar(study_levels.astype(str) + "시간", avg_changes, color=colors, alpha=0.7, width=0.5)

# 기준선 0 표시
ax.axhline(0, color='gray', linestyle='--', linewidth=1)

# 바 위에 변화량 수치 표시 (양수는 +, 음수는 -로 표시)
for bar in bars:
    height = bar.get_height()
    label_pos = height + 0.02 if height >= 0 else height - 0.05
    ax.text(bar.get_x() + bar.get_width()/2., label_pos, f'{height:+.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# 추세션 추가
ax.plot(study_levels.astype(str) + "시간", avg_changes, color='orange', marker='o', linewidth=2, linestyle=':')

plt.title('공부 시간 패턴에 따른 한동대 학생들의 실제 성적 변화 수치 (CGPA Diff)', fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('하루 평균 공부 시간 (그룹별)', fontsize=11)
ax.set_ylabel('평균 성적 변화량 (Up / Down)', fontsize=11)
ax.grid(True, linestyle=':', alpha=0.4)
plt.tight_layout()
plt.savefig('handong_grade_change_trend.png', dpi=300)
plt.close()


# [3단계] 추론통계 직접 구현 (성적 변화량 기준) - t-검정으로 글로벌 데이터와 한동대 데이터의 평균 차이 검정
print("\n성적 변화량(CGPA_Diff) 기준 추론통계 계산 결과 : ")

school_diff_array = pd.read_sql_query("SELECT CGPA_Diff FROM SCHOOL_DATA", conn)['CGPA_Diff'].to_numpy()
n_samples = len(school_diff_array)
school_diff_mean = np.mean(school_diff_array)
school_diff_std = np.sqrt(np.sum((school_diff_array - school_diff_mean) ** 2) / (n_samples - 1))
standard_error = school_diff_std / np.sqrt(n_samples)

t_critical = 2.069 
confidence_interval = (school_diff_mean - t_critical * standard_error, school_diff_mean + t_critical * standard_error)
t_statistic = (school_diff_mean - kg_diff_mean) / standard_error

print(f" 한동대 평균 성적 변화 수치 : {school_diff_mean:+.2f}점")
print(f" 한동대 학생의 실제 성적 변화량 (95% 신뢰구간) : {confidence_interval[0]:.2f}점 ~ {confidence_interval[1]:.2f}점")
print(f" 계산한 t-통계량 : {t_statistic:.4f} (임계치 : ±{t_critical})")

if abs(t_statistic) > t_critical:
    print("[최종 판정] 한동대 학생들의 성적 변화 트렌드는 글로벌 데이터와 통계적으로 '유의미한 차이'가 있습니다.")
else:
    print("[최종 판정] 한동대 학생들의 성적 변화 트렌드는 글로벌 데이터와 통계적으로 '유사'합니다.")

conn.close()
print("\n--- 분석 종료 ---")