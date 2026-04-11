import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)

df = pd.read_csv("data/Student_data.csv")

cutoff = df["Final_CGPA"].quantile(0.85)
top15 = df[df["Final_CGPA"] >= cutoff]

top15_selected = top15[[
    "Major",
    "Study_Hours_Per_Day",
    "Sleep_Hours",
    "Attendance_Pct",
    "Social_Hours_Week",
    "Final_CGPA"
]]

print("상위 15% 일부 데이터:")
print(top15_selected.head())

print("\n전공별 평균:")
result = top15_selected.groupby("Major").mean(numeric_only=True)
print(result)



# =========================
# 성적 변화 분석 추가
# =========================

df["CGPA_Change"] = df["Final_CGPA"] - df["Previous_CGPA"]

small_change = df[(df["CGPA_Change"].abs() >= 0.1) & (df["CGPA_Change"].abs() <= 0.3)]
big_change = df[df["CGPA_Change"].abs() >= 0.5]

cutoff = small_change["Final_CGPA"].quantile(0.85)
high_stable = small_change[small_change["Final_CGPA"] >= cutoff]

decline = big_change[big_change["CGPA_Change"] < 0]
increase = big_change[big_change["CGPA_Change"] > 0]

def analyze(group, name):
    print(f"\n[{name}]")
    print(group[[
        "Study_Hours_Per_Day",
        "Sleep_Hours",
        "Attendance_Pct",
        "Social_Hours_Week"
    ]].mean())

analyze(high_stable, "꾸준히 높은 그룹")
analyze(decline, "하락 그룹")
analyze(increase, "상승 그룹")