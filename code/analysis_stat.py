import sqlite3
import os
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

# 발표용 그래프 한글 깨짐 방지 설정
plt.rcParams['font.family'] = 'Malgun Gothic'  # 윈도우용 (맥북은 'AppleGothic')
plt.rcParams['axes.unicode_minus'] = False

print("=== [7조 기말 프로젝트 통계 및 SQL 분석 시작] ===")

# =========================================================================
# [1단계] SQLite 데이터베이스(DB) 구축 및 연동
# =========================================================================
print("\n[1단계] SQLite 데이터베이스 구축 중...")

# 1. student_project.db 라는 이름의 데이터베이스 파일 생성 및 연결
conn = sqlite3.connect('student_project.db')
cursor = conn.cursor()

# 2. 기존에 가공해둔 5,000명 캐글 데이터 로드 (경로는 본인 환경에 맞게 수정)
kaggle_file = 'data/Student_data_processed.csv'
df_kaggle = pd.read_csv(kaggle_file)

# 3. 우리 학교 학생 20명 데이터 로드
school_file = 'data/School_data.csv'
if os.path.exists(school_file):
    df_school = pd.read_csv(school_file)
else:
    # 만약 파일이 아직 없다면 에러 방지용 가상 데이터 20명 생성
    print("⚠️ School_data.csv 파일이 없어 임시 데이터를 생성합니다.")
    np.random.seed(7)
    df_school = pd.DataFrame({
        'ID': [f'S_{i:02d}' for i in range(1, 21)],
        'Study': np.random.uniform(2.0, 7.0, 20),
        'F_CGPA': np.random.uniform(2.5, 4.3, 20)
    })

# 4. 데이터프레임을 SQLite DB의 테이블로 저장 (엑셀 시트처럼 넣는 작업)
df_kaggle.to_sql('KAGGLE_DATA', conn, if_exists='replace', index=False)
df_school.to_sql('SCHOOL_DATA', conn, if_exists='replace', index=False)
print("✅ SQLite DB 적재 완료 (테이블명: KAGGLE_DATA, SCHOOL_DATA)")

# 5. [교수님 어필용] SQL 쿼리문으로 데이터 다시 소환하기
# "캐글 데이터 테이블에서 ID, 공부시간, 학점만 정렬해서 가져와줘"
query = "SELECT ID, Study, F_CGPA FROM KAGGLE_DATA LIMIT 5"
df_from_sql = pd.read_sql(query, conn)
print("\n💡 [SQL 실행 결과 예시] DB에서 쿼리로 불러온 데이터 상위 5행:")
print(df_from_sql)


# =========================================================================
# [2단계] 확률분포를 활용한 데이터 분석 및 해석
# =========================================================================
print("\n[2단계] 확률분포 분석 진행 중...")

# 캐글 5,000명 데이터의 '하루 공부 시간(Study)' 분포 요약
study_mean = df_kaggle['Study'].mean()
study_std = df_kaggle['Study'].std()

print(f"📊 캐글 대규모 데이터 공부 시간 평균: {study_mean:.2f}시간")
print(f"📊 캐글 대규모 데이터 공부 시간 표준편차: {study_std:.2f}시간")

# 분포 형태 시각화 및 그림 파일로 저장
plt.figure(figsize=(10, 5))
# 히스토그램과 밀도 곡선(KDE) 그리기
sns.histplot(df_kaggle['Study'], kde=True, color='skyblue', stat='density')

# 이론적인 정규분포 곡선 겹쳐 그리기 (비교용)
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = stats.norm.pdf(x, study_mean, study_std)
plt.plot(x, p, 'r--', linewidth=2, label='이론적 정규분포 선')

plt.title('대학생 하루 공부 시간의 확률분포 형태 (N=5,000)', fontsize=14)
plt.xlabel('하루 공부 시간 (시간)')
plt.ylabel('밀도 (Density)')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)

# 주피터 노트북이 아니므로 파일로 저장해야 확인 가능합니다!
plt.savefig('study_distribution.png', dpi=300)
plt.close()
print("✅ 확률분포 그래프 저장 완료! 파일명: study_distribution.png")


# =========================================================================
# [3단계] 추론통계: 신뢰구간 추정 및 가설검정
# =========================================================================
print("\n[3단계] 추론통계 분석 진행 중...")

# --- (1) 우리 학교 학생 20명의 평균 공부 시간에 대한 95% 신뢰구간 추정 ---
school_study = df_school['Study']
n_samples = len(school_study)
school_mean = school_study.mean()
# 소표본(20명)이므로 t-분포를 사용한 오차 범위 계산
school_sem = stats.sem(school_study) # 표준오차

# 95% 신뢰구간 계산
confidence_interval = stats.t.interval(0.95, df=n_samples-1, loc=school_mean, scale=school_sem)

print(f"📍 우리 학교 학생 샘플(20명)의 공부 시간 평균: {school_mean:.2f}시간")
print(f"📍 우리 학교 학생 전체의 진짜 평균 공부 시간 95% 신뢰구간: {confidence_interval[0]:.2f}시간 ~ {confidence_interval[1]:.2f}시간")


# --- (2) 가설검정 (단일표본 t-검정, One-sample t-test) ---
# 귀무가설(H0): 우리 학교 학생들의 평균 공부 시간은 캐글 대규모 트렌드(모평균)와 같다.
# 대립가설(H1): 우리 학교 학생들의 평균 공부 시간은 캐글 대규모 트렌드(모평균)와 다르다.

t_statistic, p_value = stats.ttest_1samp(school_study, popmean=study_mean)

print(f"\n🔬 [가설검정 결과]")
print(f" - t-통계량 (차이의 크기): {t_statistic:.4f}")
print(f" - p-value (우연히 일어날 확률): {p_value:.4f}")

if p_value < 0.05:
    print("🎯 결론: p-value가 0.05보다 작으므로 [귀무가설 기각]!")
    print("   -> 즉, 우리 학교 학생들의 공부 시간은 일반적인 대학생 대규모 트렌드와 통계적으로 '유의미한 차이'가 있습니다.")
else:
    print("🎯 결론: p-value가 0.05보다 크거나 같으므로 [귀무가설 채택]!")
    print("   -> 즉, 우리 학교 학생들의 공부 시간은 일반적인 대학생 대규모 트렌드와 '비슷하다'고 볼 수 있습니다.")

# 데이터베이스 연결 종료
conn.close()
print("\n=== [모든 분석 프로세스 완료] ===")