import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, GlobalAveragePooling2D, Input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array

train_dir = './forest_fire/Training and Validation'
test_dir = './forest_fire/Testing'

original_datagen = ImageDataGenerator(rescale=1./255)  # 각 픽셀을/로 스케일링

augmented_datagen = ImageDataGenerator( # 이미지 데이터를 실시간으로 증강하고 전처리
    rescale=1./255,
    rotation_range=40,  #이미지를 무작위로 최대 40도까지 회전
    width_shift_range=0.2,  #이미지를 가로 방향으로 무작위로 최대 20%까지 이동
    height_shift_range=0.2, #이미지를 세로 방향으로 무작위로 최대 20%까지 이동
    shear_range=0.2,    #이미지를 무작위로 최대 20%까지 기울기 변환
    zoom_range=0.2, #이미지를 무작위로 최대 20%까지 확대 또는 축소
    horizontal_flip=True,   #이미지를 수평으로 뒤집습니다.
    fill_mode='nearest' #변환 중 이미지의 빈 공간을 어떻게 채울지 지정. 'nearest'는 가장 가까운 픽셀 값을 사용하여 빈 공간을 채운다
)

batch_size = 32 #한 번에 생성할 이미지 배치의 크기를 32로 설정
img_height, img_width = 224, 224  # 이미지의 높이와 너비를 각각 224로 설정

original_train_generator = original_datagen.flow_from_directory(
    train_dir,  # 원본 이미지가 저장된 디렉토리 경로를 지정
    target_size=(img_height, img_width),    # 이미지를 224x224 크기로 조정
    batch_size=batch_size,  # 한 번에 생성할 이미지 배치의 크기를 32로 설정
    class_mode='binary',    # 클래스 모드를 'binary'로 설정
    color_mode='rgb',   # 이미지를 RGB 모드로 로드
    shuffle=True    # 이미지를 무작위로 섞어서 배치를 생성
)

augmented_train_generator = augmented_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary',
    color_mode='rgb',
    shuffle=True
)

def create_dataset(generator):
    dataset = tf.data.Dataset.from_generator(
        lambda: generator,
        output_types=(tf.float32, tf.float32),
        output_shapes=([None, img_height, img_width, 3], [None])
    )
    dataset = dataset.unbatch().batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

original_dataset = create_dataset(original_train_generator)
augmented_dataset = create_dataset(augmented_train_generator)

combined_train_dataset = original_dataset.concatenate(augmented_dataset)

# Test data generator
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary',
    color_mode='rgb',
    shuffle=False
)

test_dataset = create_dataset(test_generator)

