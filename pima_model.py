import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from pathlib import Path

np.random.seed(42)
tf.random.set_seed(42)

# 데이터 읽기
BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / 'diabetes.csv'
MODEL_PATH = BASE_DIR / 'pima_model.keras'
PLOT_PATH = BASE_DIR / 'training_metrics.png'

data = pd.read_csv(CSV_PATH, sep=',')

# 입력/정답 분리
X = data.values[:, 0:8]
y = data.values[:, 8]

# 정규화
scaler = MinMaxScaler()
scaler.fit(X)
X = scaler.transform(X)

# 학습/테스트 분리
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42
)

# 모델 생성
inputs = keras.Input(shape=(8,))
hidden1 = Dense(12, activation='relu')(inputs)
hidden2 = Dense(8, activation='relu')(hidden1)
output = Dense(1, activation='sigmoid')(hidden2)

model = keras.Model(inputs, output)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 학습
history = model.fit(X_train, y_train, epochs=30, batch_size=16, verbose=0)

# 학습 곡선 출력
fig, ax1 = plt.subplots()
ax1.set_xlabel('epochs')
ax1.set_ylabel('loss')
ax1.plot(history.history['loss'])

ax2 = ax1.twinx()
ax2.set_ylabel('accuracy')
ax2.plot(history.history['accuracy'])

fig.tight_layout()
fig.savefig(PLOT_PATH, dpi=150)
plt.close(fig)

# 저장
model.save(MODEL_PATH)

# 테스트 예측 확인
model = keras.models.load_model(MODEL_PATH)
X_new = X_test[:3]
print(np.round(model.predict(X_new), 2))