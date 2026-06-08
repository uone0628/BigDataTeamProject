import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('../data/Student_data_processed.csv')

# 기계학습 데이터 확인
plt.figure(figsize=(8, 6))
sns.boxplot(x='Lifestyle_Cluster', y='F_CGPA', data=df, palette='Set2')

plt.title('Lifestyle Cluster vs Final CGPA (우리 학교 학생들은 어떤 유형이 성적이 높을까?)')
plt.xlabel('Lifestyle Cluster (학생 생활 패턴 유형)')
plt.ylabel('Final CGPA (최종 성적)')

plt.show()