import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)

df = pd.read_csv("data/Student_data.csv")
df["Diff"] = df["Final_CGPA"] - df["Previous_CGPA"] 
cutoff_15 = df["Final_CGPA"].quantile(0.85)        

def display_summary(group_data, group_name):
    print(f"\n[{group_name}] 항목별 평균치")
    cols = ["Study_Hours_Per_Day", "Sleep_Hours", "Attendance_Pct", "Social_Hours_Week"]
    if not group_data.empty:
        print(group_data[cols].mean().round(2))
    else:
        print("해당 그룹에 데이터가 없습니다.")



print(" 1. 지난 성적 대비 편차에 따른 그룹 분석")

steady_high = df[
    (df["Diff"].abs().between(0.1, 0.3)) & 
    (df["Final_CGPA"] >= cutoff_15)
]

big_gap_group = df[df["Diff"].abs() >= 0.5]

increase_group = df[df["Diff"] >= 0.5]

display_summary(steady_high, "꾸준히 높은 그룹")
display_summary(big_gap_group, "지난 성적 대비 편차가 큰 그룹 (0.5 이상)")
display_summary(increase_group, "성적이 상승한 그룹 (0.5 이상 상승)")

print("\n" + "="*50 + "\n")



print(" 2. 전공 분야별 상위 15% 학생들의 생활패턴 분석")

top_15_group = df[df["Final_CGPA"] >= cutoff_15]

mission2_result = top_15_group.groupby("Major")[[
    "Final_CGPA", "Study_Hours_Per_Day", "Sleep_Hours", "Attendance_Pct", "Social_Hours_Week"
]].mean().round(2)

print(mission2_result)