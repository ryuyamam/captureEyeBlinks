import os
import sys
import cv2
import dlib
import numpy as np
import math
import datetime  # 時間
from imutils import face_utils
from scipy.spatial import distance
from PIL import Image
from overLay import overLayFaceImege as olFaceImage

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(
    'haarcascades/haarcascade_frontalface_alt2.xml')
face_parts_detector = dlib.shape_predictor(
    'haarcascades/shape_predictor_68_face_landmarks.dat')


def calc_ear(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    eye_ear = (A + B) / (2.0 * C)
    return round(eye_ear, 3)


def eye_marker(face_mat, position):
    for i, ((x, y)) in enumerate(position):
        cv2.circle(face_mat, (x, y), 1, (255, 255, 255), -1)
        cv2.putText(face_mat, str(i), (x + 2, y - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)


count = 0  # カウントめっちゃダサい
eyes_list = []
eyes_average_list = []  # 全体画像と対応
eyes_open_list = []  # 目開けている人の数を格納
eyes_close_list = []  # 目閉じている人の数を格納
faces_list = []  # 顔のlistの配列

# TODO この辺りinitの処理必要？
overLayInstamce = olFaceImage.overLayClass()  # オーバーレイクラスのインスタンス生成

while count < 20:  # TODO 撮影ボタンを押してからフレーム左フレーム分連続撮影
    tick = cv2.getTickCount()

    ret, rgb = cap.read()
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.11, minNeighbors=3, minSize=(100, 100))
    # ”a”を押したおときに処理
    if cv2.waitKey(1) == 97 or count > 0:
        if len(faces) > 0:  # 顔があると判断されたとき
            eyes_loop_list = []
            open_eyes_count = 0
            close_eyes_count = 0

            for i in range(int(len(faces))):  # 顔の数だけ
                x, y, w, h = faces[i, :]
                faces_list.append([x, y, w, h])

                cv2.rectangle(rgb, (x, y), (x + w, y + h), (255, 0, 0), 2)

                face_gray = gray[y:(y + h), x:(x + w)]
                scale = 480 / h
                face_gray_resized = cv2.resize(
                    face_gray, dsize=None, fx=scale, fy=scale)

                face = dlib.rectangle(
                    0, 0, face_gray_resized.shape[1], face_gray_resized.shape[0])
                face_parts = face_parts_detector(face_gray_resized, face)
                face_parts = face_utils.shape_to_np(face_parts)

                left_eye = face_parts[42:48]
                eye_marker(face_gray_resized, left_eye)

                left_eye_ear = calc_ear(left_eye)
                cv2.putText(rgb, "LEFT eye EAR:{} ".format(left_eye_ear),
                            (10, 100), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1, cv2.LINE_AA)

                right_eye = face_parts[36:42]
                eye_marker(face_gray_resized, right_eye)

                right_eye_ear = calc_ear(right_eye)
                cv2.putText(rgb, "RIGHT eye EAR:{} ".format(round(right_eye_ear, 3)),
                            (10, 120), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1, cv2.LINE_AA)

                # ここで(left_eye_ear + right_eye_ear)目の開きぐらいが最も大きいフレームを保存する処理
                if (left_eye_ear + right_eye_ear) < 0.5:
                    close_eyes_count += 1
                    cv2.putText(rgb, "Sleepy eyes. Wake up!",
                                (10, 180), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3, 1)
                else:
                    open_eyes_count += 1

                eyes_loop_list.append(
                    left_eye_ear + right_eye_ear)  # TODO 複数人対応をする

                # ここから付け足し
                cropped_im = rgb[y:y+h, x:x+w]
                # cv2.imwrite('output_face/outputface_'+ str(i)+str(count)+'.jpg', cropped_im)#顔id+カウント数.jpg
                cv2.imwrite('output_face/outputface_' + str(count) +
                            '.jpg', cropped_im)  # 顔id+カウント数.jpg
                cv2.imwrite('output_all/' + str(count) +
                            '.jpg', rgb)  # 顔id+カウント数.jpg
                cv2.waitKey(5)
            # ここで配列の平均値を算出
            eyes_list.append(sum(eyes_loop_list)/len(eyes_loop_list))
            eyes_open_list.append(open_eyes_count)
            eyes_close_list.append(close_eyes_count)
        count += 1  # とりあえずカウント書き方は後で直す

    fps = cv2.getTickFrequency() / (cv2.getTickCount() - tick)
    cv2.putText(rgb, "FPS:{} ".format(int(fps)),
                (10, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imshow('frame', rgb)

    if cv2.waitKey(1) == 27:
        break  # esc to quit

# 目を最も開けている画像を返す
print(eyes_list.index(max(eyes_list)))

print(eyes_open_list.index(max(eyes_open_list)))
print(eyes_close_list.index(max(eyes_close_list)))

# ベース画像の選定 imread(filename)
img_base = cv2.imread(
    'output_all/' + str(eyes_list.index(max(eyes_list)))+'.jpg')

# 顔id+カウント数.jpg
cv2.imwrite('best.jpg', img_base)
cv2.waitKey(1)

# ベスト顔画像の貼り付け
img_face = cv2.imread('output_face/outputface_' + str(eyes_list.index(min(eyes_list))) + '.jpg')
x, y, w, h = (faces_list[eyes_list.index(max(eyes_list))])

# ここからオーバーレイサンプル
# 11行目でimportしたクラスを利用して、41行目でインスタンス作成
# importしたクラス内のメソッドが利用可能になる

# 画像のオーバーレイ
# 画像の読み込み 現状1枚
image = overLayInstamce.overLayImage(img_base, (x, y))

# ウィンドウ表示
cv2.namedWindow("image", cv2.WINDOW_AUTOSIZE)
cv2.imshow("image", image)
cv2.waitKey(0)

cap.release()
cv2.destroyAllWindows()
