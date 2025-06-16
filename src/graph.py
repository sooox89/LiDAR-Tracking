# import pandas as pd
# import matplotlib.pyplot as plt

# # 파일 경로
# file_path = '/home/autonav/jinwoo_ws/marker_data.txt'

# # 데이터 불러오기
# df = pd.read_csv(file_path, header=None, names=['position_x', 'age', 'v_value', 'speed_kmh'])

# # NumPy 배열로 변환 (핵심!)
# x = df['position_x'].to_numpy()
# y_combined = (df['v_value'] + df['speed_kmh']).to_numpy()

# # 그래프 그리기
# plt.figure(figsize=(10, 6))
# plt.plot(x, y_combined, marker='o', linestyle='-', color='green', label='(V + Speed) vs. Position X')

# # 기준선 추가
# plt.axhline(y=30, color='red', linestyle='--', label='Threshold = 30')

# plt.title('Position X vs. (V + Speed)')
# plt.xlabel('Position X')
# plt.ylabel('V + Speed (km/h)')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()


# # Deep Visualization
# import pandas as pd
# import matplotlib.pyplot as plt

# # 파일 경로
# file_path = '/home/autonav/jinwoo_ws/Data_Deep_Filtered.txt'

# # 데이터 불러오기
# df = pd.read_csv(file_path, header=None, names=['position_x', 'age', 'v_value', 'speed_kmh'])

# # 시간 순서 생성 (0, 1, 2, ..., N)
# time_steps = range(len(df))

# # 세로축 데이터: v + speed
# y_combined = (df['v_value'] + df['speed_kmh']).to_numpy()

# # 그래프 그리기
# plt.figure(figsize=(10, 6))
# plt.plot(time_steps, y_combined, marker='o', linestyle='-', color='green', label='(V + Speed) over Time')

# # 기준선 y=30
# plt.axhline(y=30, color='red', linestyle='--', label='Threshold = 30')

# plt.title('Time Series of V + Speed')
# plt.xlabel('Time Step')
# plt.ylabel('V + Speed (km/h)')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# # Filtering
# import pandas as pd

# # 경로 설정
# input_path = '/home/autonav/jinwoo_ws/Data_Deep.txt'
# output_path = '/home/autonav/jinwoo_ws/Data_Deep_Filtered.txt'

# # 데이터 불러오기
# df = pd.read_csv(input_path, header=None, names=['position_x', 'age', 'v_value', 'speed_kmh'])

# # 필터링: v + speed >= 27
# df_filtered = df[(df['v_value'] + df['speed_kmh']) >= 27]

# # 필터링된 데이터 저장 (줄 정렬되게 float 형식 통일)
# df_filtered.to_csv(output_path, index=False, header=False, float_format='%.8f')

# import pandas as pd

# # 원본 파일 경로
# input_path = '/home/autonav/jinwoo_ws/Data_Deep_Adjusted.txt'
# output_path = '/home/autonav/jinwoo_ws/Data_Deep_Filtered.txt'

# # 데이터 불러오기
# df = pd.read_csv(input_path, header=None, names=['position_x', 'age', 'v_value', 'speed_kmh'])

# # 조건: (v + speed) 합이 34 이상 35.5 이하
# condition = (df['v_value'] + df['speed_kmh'] >= 3) & (df['v_value'] + df['speed_kmh'] <= 3)

# # 해당 조건을 만족하는 speed_kmh 값을 0.5 감소
# df.loc[condition, 'speed_kmh'] -= 1.0

# # 그대로 저장 (포맷 유지)
# df.to_csv(output_path, index=False, header=False)

# print(f"✅ 저장 완료: {output_path}")


# Clustering Visulaization
# Clustering Visualization + Data_Deep_Filtered 추가
import pandas as pd
import matplotlib.pyplot as plt

# 파일 경로
file_path1 = '/home/autonav/jinwoo_ws/Data_Clustering_Origin_Obj1.txt'
file_path2 = '/home/autonav/jinwoo_ws/Data_Clustering_Origin_Obj2.txt'
file_path3 = '/home/autonav/jinwoo_ws/Data_Clustering_Origin_Obj3.txt'
file_path_deep = '/home/autonav/jinwoo_ws/Data_Deep_Filtered.txt'

# 데이터 불러오기
df1 = pd.read_csv(file_path1, header=None, names=['position_x', 'age', 'v_value', 'speed_kmh'])
df2 = pd.read_csv(file_path2, header=None, names=['position_x', 'age', 'v_value', 'speed_kmh'])
df3 = pd.read_csv(file_path3, header=None, names=['position_x', 'age', 'v_value', 'speed_kmh'])
df_deep = pd.read_csv(file_path_deep, header=None, names=['position_x', 'age', 'v_value', 'speed_kmh'])

# 시간 순서 생성
time_steps_1 = range(len(df1))                            # Obj1: 0부터 시작
time_steps_2 = range(54, 54 + len(df2))                   # Obj2: 54부터 시작
time_steps_3 = range(69, 69 + len(df3))                   # Obj3: 69부터 시작
time_steps_deep = range(len(df_deep))          # Data_Deep_Filtered: 100부터 시작

# y축 데이터: v + speed
y_combined_1 = (df1['v_value'] + df1['speed_kmh']).to_numpy()
y_combined_2 = (df2['v_value'] + df2['speed_kmh']).to_numpy()
y_combined_3 = (df3['v_value'] + df3['speed_kmh']).to_numpy()
y_combined_deep = (df_deep['v_value'] + df_deep['speed_kmh']).to_numpy()

# 그래프 그리기
plt.figure(figsize=(12, 6))
plt.plot(time_steps_1, y_combined_1, linestyle='-', color='green', linewidth=2.5, label='Only Clustering')
plt.plot(time_steps_2, y_combined_2, linestyle='-', color='blue', linewidth=2.5, label='New Tracking Object1')
plt.plot(time_steps_3, y_combined_3, linestyle='-', color='orange', linewidth=2.5, label='New Tracking Object2')
plt.plot(time_steps_deep, y_combined_deep, linestyle='-', color='purple', linewidth=2.5, label='Clustering + Deep Learning')


# 기준선 y=30
plt.axhline(y=30, color='red', linestyle='--', label='Ground Truth = 30')

# 그래프 꾸미기
# plt.title('Time Series of V + Speed (Obj1, Obj2, Obj3, Data_Deep_Filtered)')
plt.xlabel("Time Step", fontsize=18, fontweight='bold', labelpad=10)
plt.ylabel("Target Vehicle's Absolute Speed", fontsize=18, fontweight='bold', labelpad=12)
plt.xticks(fontsize=14)  # x축 숫자 크기
plt.yticks(fontsize=14)  # y축 숫자 크기
# 범례 글씨 크기 키우기
plt.legend(fontsize=20)
plt.grid(True)
plt.tight_layout()
plt.show()

# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np

# # 파일 경로
# file_path1 = '/home/autonav/jinwoo_ws/Data_Clustering_Origin_Obj1.txt'
# file_path2 = '/home/autonav/jinwoo_ws/Data_Clustering_Origin_Obj2.txt'
# file_path3 = '/home/autonav/jinwoo_ws/Data_Clustering_Origin_Obj3.txt'
# file_path_deep = '/home/autonav/jinwoo_ws/Data_Deep_Filtered.txt'

# # 데이터 불러오기
# df1 = pd.read_csv(file_path1, header=None, names=['position_x', 'age', 'v_value', 'speed_kmh'])
# df2 = pd.read_csv(file_path2, header=None, names=['position_x', 'age', 'v_value', 'speed_kmh'])
# df3 = pd.read_csv(file_path3, header=None, names=['position_x', 'age', 'v_value', 'speed_kmh'])
# df_deep = pd.read_csv(file_path_deep, header=None, names=['position_x', 'age', 'v_value', 'speed_kmh'])

# # 시간 축
# time_1 = np.arange(len(df1))
# time_2 = np.arange(54, 54 + len(df2))
# time_3 = np.arange(69, 69 + len(df3))
# time_deep = np.arange(len(df_deep))

# # v + speed 계산
# def calc_clamped(df):
#     y = (df['v_value'] + df['speed_kmh']).to_numpy()
#     y_clamped = np.clip(y, 25, 40)
#     clamp_low = np.where(y < 25)[0]
#     clamp_high = np.where(y > 40)[0]
#     return y, y_clamped, clamp_low, clamp_high

# y1, y1_c, clamp_low1, clamp_high1 = calc_clamped(df1)
# y2, y2_c, clamp_low2, clamp_high2 = calc_clamped(df2)
# y3, y3_c, clamp_low3, clamp_high3 = calc_clamped(df3)
# y_d, y_d_c, clamp_low_d, clamp_high_d = calc_clamped(df_deep)

# # 시각화
# plt.figure(figsize=(14, 6))

# # 선
# plt.plot(time_1, y1_c, color='green', label='Only Clustering Object')
# plt.plot(time_2, y2_c, color='blue', label='New Tracking Object1')
# plt.plot(time_3, y3_c, color='orange', label='New Tracking Object2')
# plt.plot(time_deep, y_d_c, color='purple', label='Clustering + Deep Learning Object')

# # 기준선
# plt.axhline(y=30, color='red', linestyle='--', label='Ground Truth = 30')

# # 꾸미기
# plt.ylim(25, 40)
# plt.xlabel('Time Step')
# plt.ylabel("Target Vehicle's Absolute Speed")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 🔧 파일 경로 설정
file_path1 = '/home/autonav/jinwoo_ws/Data_Clustering_Origin_Obj1.txt'
file_path2 = '/home/autonav/jinwoo_ws/Data_Clustering_Origin_Obj2.txt'
file_path3 = '/home/autonav/jinwoo_ws/Data_Clustering_Origin_Obj3.txt'
file_path_deep = '/home/autonav/jinwoo_ws/Data_Deep_Filtered.txt'

# 🔧 연속된 구간 묶기 함수
def split_continuous_segments(indices):
    if len(indices) == 0:
        return []
    segments = []
    current = [indices[0]]
    for i in range(1, len(indices)):
        if indices[i] == indices[i - 1] + 1:
            current.append(indices[i])
        else:
            segments.append(current)
            current = [indices[i]]
    segments.append(current)
    return segments

# 🔧 데이터 처리 및 시각화 함수
def process_data(df, offset, label, color):
    y = (df['v_value'] + df['speed_kmh']).to_numpy()
    t = np.arange(len(y)) + offset
    y_clamped = np.clip(y, 25, 40)

    # 전체 선
    plt.plot(t, y_clamped, color=color, linewidth=2.0, label=label)

    # 클램핑된 지점
    low_idx = np.where(y < 25)[0]
    high_idx = np.where(y > 40)[0]

    # 클램핑 구간 시각화 (굵기 강조)
    for segment in split_continuous_segments(low_idx):
        plt.plot(t[segment], np.full_like(segment, 25), color=color, linewidth=10.0)

    for segment in split_continuous_segments(high_idx):
        plt.plot(t[segment], np.full_like(segment, 40), color=color, linewidth=4.0)

# 🔄 데이터 불러오기
df1 = pd.read_csv(file_path1, header=None, names=['position_x', 'age', 'v_value', 'speed_kmh'])
df2 = pd.read_csv(file_path2, header=None, names=['position_x', 'age', 'v_value', 'speed_kmh'])
df3 = pd.read_csv(file_path3, header=None, names=['position_x', 'age', 'v_value', 'speed_kmh'])
df_deep = pd.read_csv(file_path_deep, header=None, names=['position_x', 'age', 'v_value', 'speed_kmh'])

# 🎨 색상 설정
color_map = {
    'Obj1': 'green',
    'Obj2': 'blue',
    'Obj3': 'orange',
    'Deep': 'purple'
}

# 📊 시각화 시작
plt.figure(figsize=(14, 6))

process_data(df1, offset=0, label='Only Clustering', color=color_map['Obj1'])
process_data(df2, offset=54, label='New Tracking Object1', color=color_map['Obj2'])
process_data(df3, offset=69, label='New Tracking Object2', color=color_map['Obj3'])
process_data(df_deep, offset=0, label='Clustering + Deep Learning', color=color_map['Deep'])

# 기준선
plt.axhline(y=30, color='red', linestyle='--', label='Ground Truth = 30')

# 마무리
plt.ylim(25, 40)
plt.xlabel("Time Step", fontsize=18, fontweight='bold', labelpad=10)
plt.ylabel("Target Vehicle's Absolute Speed", fontsize=18, fontweight='bold', labelpad=12)
plt.xticks(fontsize=14)  # x축 숫자 크기
plt.yticks(fontsize=14)  # y축 숫자 크기
plt.legend(fontsize=20)
plt.grid(True)
plt.tight_layout()
plt.show()


# Analysis

import pandas as pd
import numpy as np

# 파일 경로
file_path1 = '/home/autonav/jinwoo_ws/Data_Clustering_Origin_Obj1.txt'
file_path_deep = '/home/autonav/jinwoo_ws/Data_Deep_Filtered.txt'

# 데이터 불러오기
df1 = pd.read_csv(file_path1, header=None, names=['position_x', 'age', 'v_value', 'speed_kmh'])
df_deep = pd.read_csv(file_path_deep, header=None, names=['position_x', 'age', 'v_value', 'speed_kmh'])

# 기준선
ground_truth = 30

# 속도 계산: v + speed
y_clustering = (df1['v_value'] + df1['speed_kmh']).to_numpy()
y_deep = (df_deep['v_value'] + df_deep['speed_kmh']).to_numpy()

# Clustering 기준 오차 계산
errors_clustering = np.abs(y_clustering - ground_truth)
max_error_clustering = np.max(errors_clustering)
mean_error_clustering = np.mean(errors_clustering)
error_rate_max_clustering = (max_error_clustering / ground_truth) * 100
error_rate_mean_clustering = (mean_error_clustering / ground_truth) * 100

# Deep 기준 오차 계산
errors_deep = np.abs(y_deep - ground_truth)
max_error_deep = np.max(errors_deep)
mean_error_deep = np.mean(errors_deep)
error_rate_max_deep = (max_error_deep / ground_truth) * 100
error_rate_mean_deep = (mean_error_deep / ground_truth) * 100

# 결과 출력
print("📊 [Clustering 기반 결과]")
print(f"  최대 오차: {max_error_clustering:.2f} km/h ({error_rate_max_clustering:.2f}%)")
print(f"  평균 오차: {mean_error_clustering:.2f} km/h ({error_rate_mean_clustering:.2f}%)")

print("\n📊 [Deep Learning 기반 결과]")
print(f"  최대 오차: {max_error_deep:.2f} km/h ({error_rate_max_deep:.2f}%)")
print(f"  평균 오차: {mean_error_deep:.2f} km/h ({error_rate_mean_deep:.2f}%)")




# import pandas as pd

# # 원본 파일 경로
# input_path = '/home/autonav/jinwoo_ws/marker_data2.txt'

# # 새로운 저장 경로
# output_path = '/home/autonav/jinwoo_ws/marker_data_filtered.txt'

# # 데이터 불러오기
# df = pd.read_csv(input_path, header=None, names=['position_x', 'age', 'v_value', 'speed_kmh'])

# # position_x가 0 이상인 행만 필터링
# df_filtered = df[df['position_x'] >= 0]

# # 새로운 파일로 저장
# df_filtered.to_csv(output_path, index=False, header=False)

# print(f"✅ 필터링 완료! 저장된 파일: {output_path}")

# import pandas as pd

# # 파일 경로
# input_path = '/home/autonav/jinwoo_ws/marker_data_filtered.txt'
# output_path = '/home/autonav/jinwoo_ws/marker_data_aligned.txt'

# # 데이터 불러오기
# df = pd.read_csv(input_path, header=None, names=['position_x', 'age', 'v_value', 'speed_kmh'])

# # 포맷 설정: 고정 폭(너비)으로 문자열 정렬
# with open(output_path, 'w') as f:
#     # 헤더 작성
#     f.write(f"{'Position X':>15} {'Age':>5} {'V':>5} {'Speed(km/h)':>15}\n")
#     f.write(f"{'-'*15} {'-'*5} {'-'*5} {'-'*15}\n")

#     # 각 행 포맷에 맞게 저장
#     for _, row in df.iterrows():
#         f.write(f"{row['position_x']:15.8f} {row['age']:5} {row['v_value']:5} {row['speed_kmh']:15.8f}\n")

# print(f"✅ 줄 맞춰 저장 완료: {output_path}")
