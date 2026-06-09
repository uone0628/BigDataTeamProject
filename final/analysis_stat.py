import sqlite3
import os
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.family'] = 'AppleGothic'  
plt.rcParams['axes.unicode_minus'] = False

print("--- 성적 변화 수치 추론통계 및 시각화 (최종 버전) ---")

conn = sqlite3.connect('student_project.db')

df_kaggle = pd.read_csv('data/Student_data_processed.csv')
df_school_raw = pd.read_csv('data/Handong_data.csv')

study_mapping = {'1시간 미만': 0.5, '1~2시간': 1.5, '2~3시간': 2.5, '3~4시간': 3.5, '4~5시간': 4.5, '5~6시간': 5.5, '6~7시간': 6.5, '7시간 이상': 7.5}
sleep_mapping = {'4~5시간': 4.5, '5~6시간': 5.5, '6~7시간': 6.5, '7~8시간': 7.5}
social_mapping = {'1시간 미만': 0.5, '1~2시간': 1.5, '2~3시간': 2.5, '3~4시간': 3.5, '4~5시간': 4.5, '5~6시간': 5.5, '6~7시간': 6.5, '7시간 이상': 7.5}

df_school_raw['Study'] = df_school_raw['하루 평균 공부시간(순공시간)'].map(study_mapping)
df_school_raw['Sleep'] = df_school_raw['평균 하루 수면시간 '].map(sleep_mapping)
df_school_raw['Social_Hours'] = df_school_raw['주간 평균 사교활동 시간(동아리, 새새 모임, 놀러가기 등) '].map(social_mapping)

df_school_processed = df_school_raw[['성별 ', '전공이 무엇인가요?', 'Study', 'Sleep', 'Social_Hours', '직전 학기 성적 (ex. 25-2)', '그 전학기 성적 (ex. 25-1)']].copy()
df_school_processed.columns = ['Gender', 'Major', 'Study', 'Sleep', 'Social_Hours', 'F_CGPA', 'P_CGPA']

df_school_processed['F_CGPA'] = pd.to_numeric(df_school_processed['F_CGPA'], errors='coerce')
df_school_processed['P_CGPA'] = pd.to_numeric(df_school_processed['P_CGPA'], errors='coerce')
df_school_processed = df_school_processed.dropna(subset=['Study', 'Sleep', 'Social_Hours', 'F_CGPA', 'P_CGPA'])
df_school_processed = df_school_processed[(df_school_processed['F_CGPA'] > 0) & (df_school_processed['P_CGPA'] > 0)]

df_school_processed['CGPA_Diff'] = df_school_processed['F_CGPA'] - df_school_processed['P_CGPA']

df_kaggle['Total_Hours'] = df_kaggle['Study'] + df_kaggle['Sleep'] + df_kaggle['Social_Hours']
df_kaggle['Study_Share'] = (df_kaggle['Study'] / df_kaggle['Total_Hours']) * 100

df_school_processed['Total_Hours'] = df_school_processed['Study'] + df_school_processed['Sleep'] + df_school_processed['Social_Hours']
df_school_processed['Study_Share'] = (df_school_processed['Study'] / df_school_processed['Total_Hours']) * 100

df_kaggle.to_sql('KAGGLE_DATA', conn, if_exists='replace', index=False)
df_school_processed.to_sql('SCHOOL_DATA', conn, if_exists='replace', index=False)
print("SQLite DB 적재 완료됨")


# 시각화 1 - 캐글 5,000명 성적 변화량(CGPA_Diff)의 확률분포
kaggle_diff_array = pd.read_sql_query("SELECT CGPA_Diff FROM KAGGLE_DATA", conn)['CGPA_Diff'].to_numpy()
kg_diff_mean = np.mean(kaggle_diff_array)
kg_diff_std = np.std(kaggle_diff_array, ddof=1)

plt.figure(figsize=(9, 4.5))
sns.histplot(kaggle_diff_array, kde=True, color='lightgreen', stat='density', alpha=0.6, label='실제 대규모 변화량 분포')

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



# 시각화 2 - 생활 패턴 비중(공부·수면·사교)에 따른 한동대 성적 변화 수치
query_group = """
SELECT 
    CASE 
        WHEN Study_Share < 15 THEN '1. 여가 중심형 (<15%)'
        WHEN Study_Share BETWEEN 15 AND 25 THEN '2. 생활 밸런스형 (15-25%)'
        ELSE '3. 학업 집중형 (>25%)'
    END AS Lifestyle_Pattern,
    AVG(CGPA_Diff) AS Avg_Grade_Change
FROM SCHOOL_DATA 
GROUP BY Lifestyle_Pattern
ORDER BY Lifestyle_Pattern
"""
summary_df = pd.read_sql_query(query_group, conn)

lifestyle_patterns = summary_df['Lifestyle_Pattern'].to_numpy()
avg_changes = summary_df['Avg_Grade_Change'].to_numpy()

fig, ax = plt.subplots(figsize=(9, 5))

y_min, y_max = min(avg_changes), max(avg_changes)
ax.set_ylim(y_min - 0.08, y_max + 0.1)

colors = ['#ff7675' if val < 0 else '#74b9ff' for val in avg_changes] 
bars = ax.bar(lifestyle_patterns, avg_changes, color=colors, alpha=0.8, width=0.4)

ax.axhline(0, color='gray', linestyle='--', linewidth=1)

ax.plot(lifestyle_patterns, avg_changes, color='orange', marker='o', linewidth=2, linestyle=':')

for bar in bars:
    height = bar.get_height()
    label_pos = height + 0.02 if height >= 0 else height - 0.02
    valign = 'bottom' if height >= 0 else 'top'
    
    ax.text(bar.get_x() + bar.get_width()/2., label_pos, f'{height:+.2f}', 
            ha='center', va=valign, fontsize=12, fontweight='bold',
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, pad=1.5))

plt.title('공부·수면·사교 비중 패턴에 따른 한동대 학생들의 실제 성적 변화 수치 (CGPA Diff)', fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('균형 잡힌 생활 패턴 그룹 (하루 필수 일과 중 공부 비중)', fontsize=11)
ax.set_ylabel('평균 성적 변화량 (Up / Down)', fontsize=11)
ax.grid(True, linestyle=':', alpha=0.4)
plt.tight_layout()
plt.savefig('handong_lifestyle_change_trend.png', dpi=300)
plt.close()


# [3단계] 추론통계 직접 구현 (성적 변화량 기준) - t-검정
school_diff_array = pd.read_sql_query("SELECT CGPA_Diff FROM SCHOOL_DATA", conn)['CGPA_Diff'].to_numpy()
n_samples = len(school_diff_array)
school_diff_mean = np.mean(school_diff_array)
school_diff_std = np.sqrt(np.sum((school_diff_array - school_diff_mean) ** 2) / (n_samples - 1))
standard_error = school_diff_std / np.sqrt(n_samples)

t_critical = 2.086 
confidence_interval = (school_diff_mean - t_critical * standard_error, school_diff_mean + t_critical * standard_error)
t_statistic = (school_diff_mean - kg_diff_mean) / standard_error

print(f"\n 한동대 평균 성적 변화 수치 : {school_diff_mean:+.2f}점")
print(f" 한동대 학생의 실제 성적 변화량 (95% 신뢰구간) : {confidence_interval[0]:.2f}점 ~ {confidence_interval[1]:.2f}점")
print(f" 계산한 t-통계량 : {t_statistic:.4f} (임계치 : ±{t_critical})")

if abs(t_statistic) > t_critical:
    print("결론 : 한동대 학생들의 데이터는 캐글 데이터와 통계적으로 유의미한 차이가 있음")
else:
    print("결론 : 한동대 학생들의 데이터는 캐글 데이터와 통계적으로 유사함")

conn.close()
print("\n--- 분석 종료 ---")