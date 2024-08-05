import os
import socket
import struct
import json
import time

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# 서버 IP 및 포트 설정
# server_ip = '192.168.0.96'  # 모든 인터페이스에서 연결 수락
# server_port = 12345
#
# # 소켓 생성 및 바인딩
# server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# server_sock.bind((server_ip, server_port))
# server_sock.listen(1)  # c# 클라이언트 연결 대기

################################## c++서버 연결
TCP_IP = '10.10.21.125'
TCP_PORT = 34543

TCP_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
TCP_sock.connect((TCP_IP, TCP_PORT))
#
# ##################################
# print(f"서버가 {server_port} 포트에서 클라이언트를 기다리고 있습니다...")
#
# # 클라이언트 연결 수락
# client_sock, client_addr = server_sock.accept()
# print(f"클라이언트 {client_addr} 연결 수락")
#
# while(True):
#     # 이미지 크기 수신
#     print("이미지 수신 대기")
#     data = client_sock.recv(4)
#     if len(data) < 4:
#         print("이미지 크기 수신 실패")
#         client_sock.close()
#         server_sock.close()
#         exit()
#     image_size = struct.unpack('!I', data)[0]
#     print(f"이미지 크기: {image_size}")
#
#     # 이미지 데이터 수신
#     buf = bytearray()
#     while len(buf) < image_size:
#         packet = client_sock.recv(image_size - len(buf))
#         if not packet:
#             print("이미지 데이터 수신 실패")
#             client_sock.close()
#             server_sock.close()
#             exit()
#         buf.extend(packet)
#
#     print("이미지 데이터 수신 완료")

    # 수신한 이미지를 파일로 저장
    # received_image_path = 'received_image.png'
    # with open(received_image_path, 'wb') as f:
    #     f.write(buf)
while(True):

    # 이미지 경로 및 모델 파라미터 설정
    img_height, img_width = 224, 224

    # 모델 불러오기
    model = tf.keras.models.load_model('wildfire_detection_model.keras')

    # 이미지 전처리 및 예측 함수
    def image_predict(image_path):
        # 이미지 불러오기 및 전처리
        img = load_img(image_path, target_size=(img_height, img_width))
        img_array = img_to_array(img) / 255.0  # 스케일링
        img_array = np.expand_dims(img_array, axis=0)  # 배치 차원 추가

        # 예측 수행
        prediction = model.predict(img_array)
        return prediction

    # 예측 결과 출력 함수
    def print_prediction(prediction):
        fire_probability = 1 - prediction  # "Fire" 확률
        if fire_probability >= 0.5:
            return "Fire"
        else:
            return "No Fire"

    # 예측 수행
    received_image_path = "testimage.png"
    prediction = image_predict(received_image_path)
    result = print_prediction(prediction)

    if(result == "Fire"):
        print("c++ 접속")
        first_msg = "21"
        s_msg = TCP_sock.send(first_msg.encode('utf-8'))    # c++과 프로토콜

        time.sleep(0.001)
        print("json 전송")
        result_msg = {"result_msg": prediction*100, "fire": result}
        json_msg = json.dumps(result_msg)
        TCP_sock.send(json_msg.encode())
