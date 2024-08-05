import os
import socket
import struct
import time
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# 서버 IP 및 포트 설정
server_ip = '192.168.0.96'  # 모든 인터페이스에서 연결 수락
server_port = 12345

# 소켓 생성 및 바인딩
server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_sock.bind((server_ip, server_port))
server_sock.listen(1)  # C# 클라이언트 연결 대기

# C++ 서버 연결
TCP_IP = '10.10.21.125'
TCP_PORT = 34543
TCP_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
TCP_sock.connect((TCP_IP, TCP_PORT))

print(f"서버 {server_port} 포트.")

first_msg = "21"
TCP_sock.send(first_msg.encode('utf-8'))

# 클라이언트 연결 수락
client_sock, client_addr = server_sock.accept()
print(f"클라이언트 {client_addr} 연결")

def recv_all(sock, size): # 수신 데이터 전체 읽기
    buf = bytearray()  # 가변 길이 바이트 생성
    while len(buf) < size:  # size 크기가 될 때까지 루프
        packet = sock.recv(size - len(buf))  # 읽어들인 크기에서 buf 크기만큼 빼기
        if not packet:
            return None
        buf.extend(packet)  # 읽어들인 packet을 buf에 추가, 마지막에는 size의 크기
    return buf

def recv_until(sock, delimiter):
    data = bytearray()
    while True:
        part = sock.recv(1)
        if not part:
            break
        data.extend(part)
        if data[-len(delimiter):] == delimiter:
            break
    return data

while True:
    print("이미지 크기 수신 대기")
    data = recv_all(client_sock, 4)  # 4바이트 수신 4바이트는 수신할 이미지의 크기
    if not data:
        print("이미지 크기 수신 실패")
        break

    image_size = struct.unpack('!I', data)[0]
    # !I는 빅엔디안의 4바이트 정수 의미
    # 수신된 4바이트 데이터를 정수형으로 변환하여 이미지 크기를 추출
    print(f"이미지 크기: {image_size}")

    # 이미지 데이터 수신
    print("이미지 수신 대기")
    image_data = recv_all(client_sock, image_size)
    if not image_data:
        print("이미지 데이터 수신 실패")
        break

    print("이미지 데이터 수신 완료")

    # 수신한 이미지를 파일로 저장
    received_image_path = 'received_image.jpg'
    with open(received_image_path, 'wb') as f:
        f.write(image_data)

    # 이미지 경로 및 모델 파라미터 설정
    img_height, img_width = 224, 224

    # 모델 불러오기
    model = tf.keras.models.load_model('wildfire_detection_model.keras')

    # 이미지 전처리 및 예측 함수
    def image_predict(image_path):
        img = load_img(image_path, target_size=(img_height, img_width))
        img_array = img_to_array(img) / 255.0
        # 이미지를 배열로 변환하고 픽셀 값을 0-1 범위로 정규화
        img_array = np.expand_dims(img_array, axis=0)
        # 전처리된 이미지를 모델에 넣어 예측 수행
        prediction = model.predict(img_array)
        return prediction

    # 예측 결과 출력 함수
    def print_prediction(prediction):
        # 화재 확률을 계산하기 위해 1에서 마이너스
        fire_probability = prediction
        if fire_probability >= 0.5:
            return 0
        else:
            return 1

    # 예측 수행
    prediction = image_predict(received_image_path)
    fire_pro = 1 - prediction
    result = print_prediction(fire_pro)
    print(f"prediction: {prediction}")
    print(f"probability: {fire_pro}")

    # 예측 결과 포맷팅 (소수점 두 번째 자리까지)
    format_fire = np.floor(fire_pro * 100) / 100.0
    print(format_fire)
    fire_proba = format_fire.item()
    print(f"화재확률: {fire_proba}")

    final_fire = np.floor(fire_proba * 10) / 10.0
    real_fire = final_fire * 100
    print(f"화재: {real_fire}")

    print("JSON 데이터 전송")
    time.sleep(0.5)
    result_msg = {"result_msg": real_fire, "fire": result}
    print(result_msg)

    # JSON 데이터를 C++ 서버로 전송 (예시, 실제 전송 코드는 추가해야 함)
    TCP_sock.send(str(result_msg).encode('utf-8'))
