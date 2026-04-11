import pandas as pd

# 1. 원본 데이터 불러오기
df = pd.read_csv('data/Student_data.csv')

# 2. 데이터가 어떻게 생겼는지 파악하기 (Discovering)
print("👀 1. 원본 데이터 위에서 5줄 미리보기:")
print(df.head())

print("\n-------------------------------------------")

print("📊 2. 데이터의 전체적인 정보 확인 (행 개수, 열 이름, 결측치 등):")
print(df.info())


# ==========================================
# 🍲 1. [공통 전처리] 진짜 안 쓰는 것만 버리기!
# ==========================================
df = pd.read_csv('data/Student_data.csv')

# ① 나이와 성별만 버리기 (공부, 수면, 출석률, 사교활동은 무조건 살림!)
columns_to_drop = ['Age', 'Gender'] 
df_clean = df.drop(columns=columns_to_drop)

# ② 결측치 및 이상치 제거 (Cleaning & Validating)
df_clean = df_clean.dropna() 
df_clean = df_clean[(df_clean['Sleep_Hours'] > 0) & (df_clean['Sleep_Hours'] <= 24)]
df_clean = df_clean[(df_clean['Study_Hours_Per_Day'] >= 0) & (df_clean['Study_Hours_Per_Day'] <= 24)]

# ==========================================
# 🥘 2. [방향성 1] 성적 변화에 따른 3그룹 세팅
# ==========================================
df_trend = df_clean.copy()

# 💡 여기서 1번 방향성에 필요 없는 '전공(Major)'을 버려줌!
if 'Major' in df_trend.columns:
    df_trend = df_trend.drop(columns=['Major'])

# 상위 15% 커트라인 점수 계산
top_15_val = df_trend['Final_CGPA'].quantile(0.85)

# 성적 차이 구하기
df_trend['Grade_Diff'] = df_trend['Final_CGPA'] - df_trend['Previous_CGPA']

# 3그룹 분류 함수
def categorize_with_top15(row):
    diff = row['Grade_Diff']
    cgpa = row['Final_CGPA']
    
    if diff >= 0.5:
        return '2_성적 0.5이상 상승'
    elif diff <= -0.5:
        return '3_성적 0.5이하 하락'
    elif -0.3 <= diff <= 0.3:
        if cgpa >= top_15_val:
            return '1_상위권 유지(Top 15%)'
        else:
            return '버릴 데이터'
    else:
        return '버릴 데이터'

# 이름표 붙이고 애매한 데이터 삭제
df_trend['Grade_Group'] = df_trend.apply(categorize_with_top15, axis=1)
df_trend = df_trend[df_trend['Grade_Group'] != '버릴 데이터']


# ==========================================
# 📊 3. 밥상 차리기 (기획서에 명시된 4가지 지표 요약!)
# ==========================================
print("\n👀 [1] 최종 분석 대상 그룹별 학생 수")
print(df_trend['Grade_Group'].value_counts().sort_index())
print("\n" + "="*60 + "\n")

print("📈 [2] 그룹별 4대 지표 평균 (공부, 수면, 출석률, 사교활동)")
# 기획서에 있는 4가지 지표를 리스트로 묶어서 한 번에 평균 내기!
target_columns = ['Study_Hours_Per_Day', 'Sleep_Hours', 'Attendance_Pct', 'Social_Hours_Week']

summary = df_trend.groupby('Grade_Group')[target_columns].mean()
print(summary)

# ==========================================
# 💾 4. 랭글링된 결과 파일로 저장하기 (진짜 마무리!)
# ==========================================
# 1) 요약된 밥상(평균표) 저장
summary.to_csv('data/Direction1_Summary.csv')

# 2) 나중에 그래프 그릴 때 쓸 수도 있으니 깔끔해진 전체 데이터도 저장
df_trend.to_csv('data/Direction1_Cleaned_Data.csv', index=False)
print("✨ 성공적으로 데이터를 가공하고 파일로 저장했습니다!")


import pandas as pd

# (앞부분에 df_clean을 만드는 공통 전처리 코드가 있다고 가정)

# ==========================================
# 🥘 [방향성 2] 전공 분야별 상위 15% 학생 특징 분석
# ==========================================
# 1. 훼손되지 않은 df_clean 복사 (전공 데이터 살아있음!)
df_major = df_clean.copy()

# 2. [논리 B 적용] 각 전공 '내부'에서 상위 15%만 필터링하는 함수 만들기
def filter_top_15_per_major(group):
    # 각 전공 그룹 안에서 상위 15% (하위 85%) 커트라인 점수 계산
    threshold = group['Final_CGPA'].quantile(0.85)
    # 커트라인 이상인 학생들만 반환
    return group[group['Final_CGPA'] >= threshold]

# 전공별로 묶은 뒤(groupby), 위에서 만든 필터링 함수 적용
df_top15_by_major = df_major.groupby('Major', group_keys=False).apply(filter_top_15_per_major)


# ==========================================
# 📊 3. 밥상 차리기 (전공별 우등생들의 4대 지표 요약)
# ==========================================
print("\n👀 [방향성 2] 전공별 상위 15% 학생 수 확인")
print(df_top15_by_major['Major'].value_counts())
print("\n" + "="*60 + "\n")

print("📈 전공별 상위 15% 학생들의 평균 생활 패턴")
target_columns = ['Final_CGPA', 'Study_Hours_Per_Day', 'Sleep_Hours', 'Attendance_Pct', 'Social_Hours_Week']

# 전공별로 다시 묶어서 평균 구하기
major_summary = df_top15_by_major.groupby('Major')[target_columns].mean()

# 결과물 보기 좋게 소수점 2자리까지만 반올림 (선택사항)
major_summary = major_summary.round(2)
print(major_summary)

# 파일로 저장 (필요시)
# major_summary.to_csv('data/Direction2_Summary.csv')

import pandas as pd

# 1. 원본 데이터 복사 (df_clean이 이미 있다고 가정)
df_major = df_clean.copy()
# ==========================================
# 🥘 [방향성 2] 전공 분야별 상위 15% 학생 특징 분석
# ==========================================
# 1. 훼손되지 않은 df_clean 복사 (전공 데이터 살아있음!)
df_major = df_clean.copy()

# 2. [논리 B 적용] 각 전공 '내부'에서 상위 15%만 필터링하는 함수 만들기
def filter_top_15_per_major(group):
    # 각 전공 그룹 안에서 상위 15% (하위 85%) 커트라인 점수 계산
    threshold = group['Final_CGPA'].quantile(0.85)
    # 커트라인 이상인 학생들만 반환
    return group[group['Final_CGPA'] >= threshold]

# 전공별로 묶은 뒤(groupby), 위에서 만든 필터링 함수 적용
df_top15_by_major = df_major.groupby('Major', group_keys=False).apply(filter_top_15_per_major)


# ==========================================
# 📊 3. 밥상 차리기 (전공별 우등생들의 4대 지표 요약)
# ==========================================
print("\n👀 [방향성 2] 전공별 상위 15% 학생 수 확인")
print(df_top15_by_major['Major'].value_counts())
print("\n" + "="*60 + "\n")

print("📈 전공별 상위 15% 학생들의 평균 생활 패턴")
target_columns = ['Final_CGPA', 'Study_Hours_Per_Day', 'Sleep_Hours', 'Attendance_Pct', 'Social_Hours_Week']

# 전공별로 다시 묶어서 평균 구하기
major_summary = df_top15_by_major.groupby('Major')[target_columns].mean()

# 결과물 보기 좋게 소수점 2자리까지만 반올림 (선택사항)
major_summary = major_summary.round(2)
print(major_summary)

# 파일로 저장 (필요시)
# major_summary.to_csv('data/Direction2_Summary.csv')
# ==========================================
# 📊 [대안 A] 전공별 성적 격차(Gap) 분석 (에러 수정본)
# ==========================================
def analyze_gap(group):
    # 각 전공 내부의 상위 15%와 하위 15% 커트라인 계산
    top_th = group['Final_CGPA'].quantile(0.85)
    bot_th = group['Final_CGPA'].quantile(0.15)
    
    # 상위 15% 그룹 평균과 하위 15% 그룹 평균 구하기
    top_avg = group[group['Final_CGPA'] >= top_th].mean()
    bot_avg = group[group['Final_CGPA'] <= bot_th].mean()
    
    # 격차(Gap) 반환
    return top_avg - bot_avg

# 분석할 수치 컬럼 지정
numeric_cols = ['Final_CGPA', 'Study_Hours_Per_Day', 'Sleep_Hours', 'Attendance_Pct', 'Social_Hours_Week']

# 그룹바이 실행 (에러 방지를 위해 numeric_cols를 미리 지정)
gap_analysis = df_major.groupby('Major')[numeric_cols].apply(analyze_gap)

# 보고서용 4대 지표만 추출해서 예쁘게 출력
gap_report = gap_analysis[['Study_Hours_Per_Day', 'Attendance_Pct', 'Sleep_Hours', 'Social_Hours_Week']]
print("🔥 [대안 A] 전공별 성적 상/하위 15% 격차")
print(gap_report.round(2).sort_values(by='Study_Hours_Per_Day', ascending=False))


# (df_major와 sleep_threshold가 이미 있다고 가정)

# ==========================================
# 🧟 [대안 B-2] 전공별 수면 부족 학생 '비율(%)' 분석
# ==========================================
# 1. 각 전공별 '전체' 학생 수 구하기
total_students_per_major = df_major['Major'].value_counts()

# 2. 각 전공별 '수면 부족(좀비)' 학생 수 구하기 (아까 필터링한 df_zombies 재활용)
sleep_threshold = df_major['Sleep_Hours'].quantile(0.15)
df_zombies = df_major[df_major['Sleep_Hours'] <= sleep_threshold]
zombie_counts = df_zombies['Major'].value_counts()

# 3. 데이터프레임으로 합치기
zombie_ratio_df = pd.DataFrame({
    '전체_학생수': total_students_per_major,
    '수면부족_학생수': zombie_counts,
    '이들의_평균학점': df_zombies.groupby('Major')['Final_CGPA'].mean()
})

# 4. 비율(%) 계산하기 (좀비 수 / 전체 수 * 100)
zombie_ratio_df['좀비_비율(%)'] = (zombie_ratio_df['수면부족_학생수'] / zombie_ratio_df['전체_학생수']) * 100

# 5. 보기 좋게 정리: 비율(%)이 높은 순으로 내림차순 정렬!
zombie_ratio_df = zombie_ratio_df.sort_values(by='좀비_비율(%)', ascending=False).round(2)

print("📊 [대안 B-2] 전공별 수면 부족 학생 '비율' 팩트 체크")
print(f"기준: 수면 시간 {sleep_threshold:.1f}시간 이하")
print("-" * 60)
# 보고서에 넣기 딱 좋은 컬럼 순서로 출력
print(zombie_ratio_df[['전체_학생수', '수면부족_학생수', '좀비_비율(%)', '이들의_평균학점']])