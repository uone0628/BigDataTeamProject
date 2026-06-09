import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import platform

# ============================================================
# 한글 폰트 깨짐 방지 설정
# ============================================================
if platform.system() == 'Darwin': # Mac 환경
    plt.rc('font', family='AppleGothic')
elif platform.system() == 'Windows': # Windows 환경
    plt.rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False # 마이너스 기호 깨짐 방지

# ============================================================
# 1. 데이터 불러오기 및 준비
# ============================================================
print("데이터를 불러오고 있습니다...")
df = pd.read_csv('data/Student_data.csv')

df.rename(columns={
    'Study_Hours_Per_Day': 'Study',
    'Sleep_Hours': 'Sleep',
    'Social_Hours_Week': 'Social_Hours',
    'Attendance_Pct': 'Attendance',
    'Final_CGPA': 'F_CGPA',
}, inplace=True)

study_hours = df['Study'].values
sleep_hours = df['Sleep'].values
social_hours = df['Social_Hours'].values
attendance = df['Attendance'].values
score = df['F_CGPA'].values

# ============================================================
# 2. 다중 선형 회귀 모델 학습
# ============================================================
independent_variable_matrix = np.column_stack([
    study_hours, sleep_hours, social_hours, attendance, np.ones_like(study_hours),
])

coef_study, coef_sleep, coef_social, coef_atten, intercept = np.linalg.lstsq(
    independent_variable_matrix, score, rcond=None,
)[0]

score_pred = (coef_study * study_hours + 
              coef_sleep * sleep_hours + 
              coef_social * social_hours + 
              coef_atten * attendance + intercept)

# ============================================================
# 3. 모델 평가
# ============================================================
error = score - score_pred
mae = np.mean(np.abs(error))
rmse = np.sqrt(np.mean(error ** 2))
residual_sum = np.sum((score - score_pred) ** 2)
total_sum = np.sum((score - np.mean(score)) ** 2)
r2 = 1.0 - residual_sum / total_sum

equation_text = (
    f"예상 학점 = {coef_study:.4f} * 공부시간\n"
    f"              + {coef_sleep:.4f} * 수면시간\n"
    f"              + {coef_social:.4f} * 사교시간\n"
    f"              + {coef_atten:.4f} * 출석률\n"
    f"              + {intercept:.4f} (기본점수)"
)

print("\n[다중 선형 회귀 분석 결과: 생활습관으로 학점 예측]")
print(equation_text)
print(f"MAE (평균 오차) = {mae:.4f}")
print(f"R2 (설명력)     = {r2:.4f}")

# ============================================================
# 4. 결과 시각화 (실제값 vs 예측값)
# ============================================================
fig, ax_result = plt.subplots(figsize=(8, 6))

ax_result.scatter(score, score_pred, alpha=0.5, color="tab:green")

min_value = min(score.min(), score_pred.min())
max_value = max(score.max(), score_pred.max())
padding = max((max_value - min_value) * 0.05, 1.0)
lower = min_value - padding
upper = max_value + padding

ax_result.plot([lower, upper], [lower, upper], color="tab:red", linewidth=2)
ax_result.set_xlim(lower, upper)
ax_result.set_ylim(lower, upper)
ax_result.set_aspect("equal", adjustable="box")
ax_result.set_title("실제 학점 vs 예측한 학점", fontsize=14, fontweight='bold')
ax_result.set_xlabel("실제 학점 (점)", fontsize=12)
ax_result.set_ylabel("예측 학점 (점)", fontsize=12)
ax_result.grid(alpha=0.25)

result_text = (
    f"MAE (오차) = {mae:.4f}\n"
    f"RMSE = {rmse:.4f}\n"
    f"R2 (설명력) = {r2:.4f}"
)
ax_result.text(
    0.04, 0.96, result_text, transform=ax_result.transAxes,
    va="top", bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9}, fontsize=11
)

plt.show()