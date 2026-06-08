import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore') # 경고 메시지 숨기기

def load_and_clean_data(filepath):
    print("1. 데이터 불러오기 및 기본 컬럼명 정리 단계")
    # 데이터 로드
    df = pd.read_csv(filepath)

    # 컬럼명 간소화 (기존 코드 유지)
    df.rename(columns={
        'Student_ID': 'ID',
        'Attendance_Pct': 'Attendance',
        'Study_Hours_Per_Day': 'Study',
        'Previous_CGPA': 'P_CGPA',
        'Sleep_Hours': 'Sleep',
        'Social_Hours_Week': 'Social_Hours',
        'Final_CGPA': 'F_CGPA',
    }, inplace=True)
    
    return df

def create_advanced_features(df):
    print("2. 기계학습(군집화)을 활용한 새로운 컬럼 생성 단계")
    
    # K-Means 군집화를 활용한 컬럼 생성 - 학생 생활습관 유형(Lifestyle_Cluster)
    features_for_clustering = ['Study', 'Sleep', 'Social_Hours', 'Attendance']
    
    # 변수들마다 단위(시간, % 등)가 다르므로 스케일링(표준화) 작업
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df[features_for_clustering])
    
    # K-Means 모델 학습 (학생들을 3가지 유형으로 자동 분류)
    kmeans = KMeans(n_clusters=3, random_state=42)
    df['Lifestyle_Cluster'] = kmeans.fit_predict(scaled_features)
    
    # 클러스터 번호(0, 1, 2)에 임시로 네이밍 - 추후 시각화 이후 이름 지정 ㄱㄱ
    cluster_map = {0: 'Type_A', 1: 'Type_B', 2: 'Type_C'}
    df['Lifestyle_Cluster'] = df['Lifestyle_Cluster'].map(cluster_map)

    # 분석을 위한 타겟 변수 생성 (이전 성적 대비 올랐는지 내렸는지 파악용)
    df['CGPA_Diff'] = df['F_CGPA'] - df['P_CGPA']
    
    return df

def save_processed_data(df, output_path):
    print("3. 컬럼 순서 재배치 및 최종 데이터 저장 과정")
    
    # 컬럼 순서 재배치
    new_column_order = [
        'ID', 'Gender', 'Age', 'Major',
        'Sleep', 'Study', 'Social_Hours', 'Attendance',
        'Lifestyle_Cluster',
        'P_CGPA', 'F_CGPA', 'CGPA_Diff'
    ]
    df = df[new_column_order]
    
    # 최종 파일 저장
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"✅ 데이터 랭글링 완료! 파일 저장 위치: {output_path}")
    return df

# 메인 실행 함수
if __name__ == "__main__":
    input_filepath = 'data/Student_data.csv'  
    output_filepath = 'data/Student_data_processed.csv'
    
    df_cleaned = load_and_clean_data(input_filepath)
    df_featured = create_advanced_features(df_cleaned)
    df_final = save_processed_data(df_featured, output_filepath)
    
    print("\n--- 랭글링 완료 ---")
    print(df_final.head())