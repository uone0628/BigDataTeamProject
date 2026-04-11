import pandas as pd

pd.set_option("display.max_columns", None)
df = pd.read_csv("data/Student_data.csv")

# ==========================================
# 1. 이전 성적과 편차 분석
# ==========================================
print("\n[1. 이전 성적과 편차 분석]")

df["GPA_Diff"] = df["Final_CGPA"] - df["Previous_CGPA"]
df["GPA_Gap_Size"] = df["GPA_Diff"].abs()

low_gap_df = df[df["GPA_Gap_Size"].between(0.1, 0.3)]
high_gap_df = df[df["GPA_Gap_Size"] >= 0.5]

top_limit = low_gap_df["Final_CGPA"].quantile(0.85)
steady_ranker = low_gap_df[low_gap_df["Final_CGPA"] >= top_limit]

group_down = high_gap_df[high_gap_df["GPA_Diff"] < 0]
group_up = high_gap_df[high_gap_df["GPA_Diff"] > 0]

def show_habit_stats(target_df, title):
    print(f"\n- {title}")
    features = ["Study_Hours_Per_Day", "Sleep_Hours", "Attendance_Pct", "Social_Hours_Week"]
    print(target_df[features].mean(numeric_only=True).round(2))

show_habit_stats(steady_ranker, "꾸준히 높은 그룹 (편차 0.1~0.3)")
show_habit_stats(group_down, "하락한 그룹 (편차 0.5 이상)")
show_habit_stats(group_up, "상승한 그룹 (편차 0.5 이상)")

# ==========================================
# 2. 전공 분야별 높은 성적군의 특징
# ==========================================
print("\n\n[2. 전공 분야별 높은 성적군의 특징]")

major_top_group = df.groupby("Major").apply(
    lambda x: x[x["Final_CGPA"] >= x["Final_CGPA"].quantile(0.85)]
).reset_index(drop=True)

major_summary = major_top_group.groupby("Major")[[
    "Study_Hours_Per_Day", 
    "Sleep_Hours", 
    "Attendance_Pct", 
    "Social_Hours_Week",
    "Final_CGPA"
]].mean(numeric_only=True).round(2)

print("\n전공별 상위 15% 평균 지표:")
print(major_summary)

