import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
import platform

# ============================================================
# 한글 폰트 깨짐 방지 설정
# ============================================================
if platform.system() == 'Darwin':
    plt.rc('font', family='AppleGothic')
elif platform.system() == 'Windows':
    plt.rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 1. 기본 설정 및 데이터 준비
# ============================================================
MAX_DEPTH_CANDIDATES = np.arange(1, 8)
MIN_SAMPLES_LEAF = 5
RANDOM_STATE = 7

# 트리에 표시될 변수명과 등급명 한글화
FEATURE_NAMES = ["공부시간", "수면시간", "사교시간", "출석률"]
CLASS_NAMES = ["하위권(C)", "중위권(B)", "상위권(A)"]

df = pd.read_csv('data/Student_data.csv')

def categorize_grade(cgpa):
    if cgpa < 2.5: return 0    # 하위권
    elif cgpa < 3.5: return 1  # 중위권
    else: return 2             # 상위권

df['Grade_Class'] = df['Final_CGPA'].apply(categorize_grade)

features_all = df[['Study_Hours_Per_Day', 'Sleep_Hours', 'Social_Hours_Week', 'Attendance_Pct']].values
skill_level_all = df['Grade_Class'].values

X_temp, features_test, y_temp, skill_level_test = train_test_split(
    features_all, skill_level_all, test_size=0.2, random_state=RANDOM_STATE
)
features_train, features_validation, skill_level_train, skill_level_validation = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=RANDOM_STATE
)

# ============================================================
# 2. 결정 트리 모델 학습 및 검증
# ============================================================
train_error_history = []
validation_error_history = []
best_validation_error = np.inf
best_depth = None
best_tree_model = None

for max_depth in MAX_DEPTH_CANDIDATES:
    candidate_tree_model = DecisionTreeClassifier(
        max_depth=max_depth, min_samples_leaf=MIN_SAMPLES_LEAF, random_state=RANDOM_STATE
    )
    candidate_tree_model.fit(features_train, skill_level_train)

    train_pred = candidate_tree_model.predict(features_train)
    validation_pred = candidate_tree_model.predict(features_validation)
    train_error = 1 - np.mean(train_pred == skill_level_train)
    validation_error = 1 - np.mean(validation_pred == skill_level_validation)

    train_error_history.append(train_error)
    validation_error_history.append(validation_error)

    if validation_error < best_validation_error:
        best_validation_error = validation_error
        best_depth = max_depth
        best_tree_model = candidate_tree_model

tree_model = best_tree_model
skill_level_pred = tree_model.predict(features_test)
accuracy = np.mean(skill_level_pred == skill_level_test)

print(f"\n[결정 트리 분류 결과]")
print(f"최적의 트리 깊이: {best_depth}")
print(f"테스트 정답률(정확도): {accuracy * 100:.2f}%\n")

# ============================================================
# 3. 결과 시각화 (박스 겹침 해결)
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(18, 7), gridspec_kw={'width_ratios': [1, 2.5]}, constrained_layout=True)

axes[0].plot(MAX_DEPTH_CANDIDATES, train_error_history, marker="o", color="tab:purple", label="학습 데이터 오차")
axes[0].plot(MAX_DEPTH_CANDIDATES, validation_error_history, marker="o", color="tab:orange", label="검증 데이터 오차")
axes[0].axvline(best_depth, color="tab:red", linestyle="--", label="최적 깊이")
axes[0].set_title("트리 깊이에 따른 오차 변화", fontsize=14, fontweight='bold')
axes[0].set_xlabel("트리 최대 깊이 (max_depth)", fontsize=12)
axes[0].set_ylabel("오차율 (Error)", fontsize=12)
axes[0].legend(loc="best")
axes[0].grid(alpha=0.25)

plot_tree(
    tree_model,
    feature_names=FEATURE_NAMES,
    class_names=CLASS_NAMES,
    filled=True,
    rounded=True,
    impurity=False,
    fontsize=7,  
    ax=axes[1],
)
axes[1].set_title("상위권(A등급) 구하는 이진 트리", fontsize=15, fontweight='bold')

plt.savefig('Decision_Tree_Result.png', dpi=300, bbox_inches='tight')

plt.show()