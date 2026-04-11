import pandas as pd

# ==========================================
# 0. 데이터 불러오기 및 기본 탐색 (Discovering)
# ==========================================
df = pd.read_csv('data/Student_data.csv')

# 👀 [PM님 픽!] 데이터가 어떻게 생겼는지 파악하기
print("👀 1. 원본 데이터 위에서 5줄 미리보기:")
print(df.head())
print("\n" + "-" * 43 + "\n")
print("📊 2. 데이터의 전체적인 정보 확인 (행 개수, 열 이름, 결측치 등):")
print(df.info())
print("\n" + "=" * 60 + "\n")

# 기초 전처리: 불필요 컬럼 제거 및 결측치/이상치 처리
df_clean = df.drop(columns=['Age', 'Gender']).dropna()
df_clean = df_clean[(df_clean['Sleep_Hours'] > 0) & (df_clean['Sleep_Hours'] <= 24)]
df_clean = df_clean[(df_clean['Study_Hours_Per_Day'] >= 0) & (df_clean['Study_Hours_Per_Day'] <= 24)]


# ==========================================
# 1. [방향성 1] 성적 변화 그룹 분석 (0.4 제외 로직)
# ==========================================
df_trend = df_clean.copy()
top_15_val = df_trend['Final_CGPA'].quantile(0.85)
df_trend['Grade_Diff'] = df_trend['Final_CGPA'] - df_trend['Previous_CGPA']

def categorize_trend(row):
    diff = row['Grade_Diff']
    cgpa = row['Final_CGPA']
    
    if diff >= 0.5:
        return '2_성적 상승(0.5↑)'
    elif diff <= -0.5:
        return '3_성적 하락(-0.5↓)'
    elif -0.3 <= diff <= 0.3:
        if cgpa >= top_15_val:
            return '1_상위권 유지(Top 15%)'
    return '제외 데이터' 

df_trend = df_trend[df_trend.apply(categorize_trend, axis=1) != '제외 데이터']
df_trend['Grade_Group'] = df_trend.apply(categorize_trend, axis=1)

target_cols = ['Study_Hours_Per_Day', 'Sleep_Hours', 'Attendance_Pct', 'Social_Hours_Week']
trend_summary = df_trend.groupby('Grade_Group')[target_cols].mean().round(2)
print("\n📊 [방향성 1] 성적 변화 그룹별 평균 지표")
print(trend_summary)


# ==========================================
# 2. [방향성 2] 전공별 분석 (평균표 + 심화 분석)
# ==========================================
df_major = df_clean.copy()
target_columns = ['Final_CGPA', 'Study_Hours_Per_Day', 'Sleep_Hours', 'Attendance_Pct', 'Social_Hours_Week']

# (1) 전공별 상위 15% 학생들의 단순 평균표
def get_top_15(group):
    return group[group['Final_CGPA'] >= group['Final_CGPA'].quantile(0.85)]

major_top15 = df_major.groupby('Major', group_keys=False).apply(get_top_15)
major_avg_summary = major_top15.groupby('Major')[target_columns].mean().round(2)
print("\n📈 [방향성 2-기본] 전공별 상위 15% 평균 생활 패턴")
print(major_avg_summary)

# (2) 상/하위 성적 격차(Gap) 분석
def analyze_gap(group):
    top_avg = group[group['Final_CGPA'] >= group['Final_CGPA'].quantile(0.85)].mean(numeric_only=True)
    bot_avg = group[group['Final_CGPA'] <= group['Final_CGPA'].quantile(0.15)].mean(numeric_only=True)
    return top_avg - bot_avg

gap_report = df_major.groupby('Major')[target_columns].apply(analyze_gap)
print("\n🔥 [방향성 2-심화] 전공별 성적 상/하위 15% 격차 (Gap)")
print(gap_report[['Study_Hours_Per_Day', 'Attendance_Pct']].round(2))

# (3) 수면 부족 학생 비율 분석
sleep_threshold = df_major['Sleep_Hours'].quantile(0.15)
df_zombies = df_major[df_major['Sleep_Hours'] <= sleep_threshold]
zombie_ratio = (df_zombies['Major'].value_counts() / df_major['Major'].value_counts() * 100).fillna(0)

print(f"\n🧟 [방향성 2-심화] 전공별 수면 부족 학생 비율 (기준: {sleep_threshold:.1f}시간)")
print(zombie_ratio.sort_values(ascending=False).round(2))