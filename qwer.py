import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import csv
from datetime import datetime
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator


file_path = r'C:\Users\kangm\Downloads\S01_PPG.csv'
data = pd.read_csv(file_path, encoding='latin1', sep=',', on_bad_lines='skip', engine='python')
data.columns = data.columns.str.strip()
heart_rate_data = data[['PPG0', 'PPG1', 'PPG2']].mean(axis=1).values  # 평균 심박수 데이터


def adjust_heart_rate(data, scale_min=50, scale_max=150):
    data_min, data_max = data.min(), data.max()
    adjusted = (data - data_min) / (data_max - data_min)  # 0~1로 정규화
    adjusted = adjusted * (scale_max - scale_min) + scale_min  # 스케일 조정
    return adjusted


def is_abnormal_heart_rate(heart_rate):
    normal_range = (60, 100)  # 정상 심박수 범위 정의
    return heart_rate < normal_range[0] or heart_rate > normal_range[1]


adjusted_heart_rate_data = adjust_heart_rate(heart_rate_data)


scaler = MinMaxScaler()
heart_rate_data_scaled = scaler.fit_transform(adjusted_heart_rate_data.reshape(-1, 1))


time_steps = 10
generator = TimeseriesGenerator(heart_rate_data_scaled, heart_rate_data_scaled, length=time_steps, batch_size=32)


model = Sequential([
    LSTM(50, activation='relu', input_shape=(time_steps, 1)),
    Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error')


model.fit(generator, epochs=20, verbose=1)


def send_alert(heart_rate):
    print(f"경고: 비정상 심박수 감지! 심박수: {heart_rate:.2f} BPM. 관리자가 확인이 필요합니다.")


def log_event(heart_rate, is_abnormal):
    with open("heart_rate_log.csv", mode="a", newline="") as file:
        writer = csv.writer(file)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        writer.writerow([timestamp, heart_rate, is_abnormal])


def plot_heart_rate(actual_data):
    plt.cla()
    plt.plot(actual_data, label='실제 심박수', color='blue')
    plt.title("실시간 심박수 변화")
    plt.xlabel("시간 (초)")
    plt.ylabel("심박수 (BPM)")
    plt.legend()
    plt.pause(0.1)


collected_heart_rate = []
abnormal_count = 0

plt.ion()
for i, heart_rate in enumerate(adjusted_heart_rate_data):
    collected_heart_rate.append(heart_rate)
    print(f"보정된 심박수: {heart_rate:.2f} BPM")


    if len(collected_heart_rate) >= time_steps:
        heart_rate_sequence = np.array(collected_heart_rate[-time_steps:]).reshape(-1, 1)
        heart_rate_sequence_scaled = scaler.transform(heart_rate_sequence)
        heart_rate_sequence_scaled = heart_rate_sequence_scaled.reshape((1, time_steps, 1))
        predicted_heart_rate = model.predict(heart_rate_sequence_scaled, verbose=0)[0][0]
        predicted_heart_rate = scaler.inverse_transform([[predicted_heart_rate]])[0][0]


        if is_abnormal_heart_rate(predicted_heart_rate):
            abnormal_count += 1
            send_alert(predicted_heart_rate)


        log_event(predicted_heart_rate, is_abnormal_heart_rate(predicted_heart_rate))


    if abnormal_count >= 5:
        print("경고: 비정상 심박수가 5회 이상 지속되었습니다!")


    plot_heart_rate(collected_heart_rate)


    if i == len(adjusted_heart_rate_data) - 1:
        break

plt.ioff()
plt.show()
