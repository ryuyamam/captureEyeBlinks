#-完了-
#顔認識
#画像のリサイズと貼り付け位置
#画像の貼り付け完了12/09
#複数人対応完了12/09
#顔の回転に対応させる(はるき)12/09
#読み込んだ人数が最も多い　かつ　最も最善な画像を選定　って処理にする12/09

#-未完了-
#60fpsで撮影して保存→撮影後に再度読み込むところまで(ヤンキー)
#撮影中の顔id対応させる(ヤンキー去年やったんやからやれ)
#ジャンプしたときをフレーム選定する処理(りゅうやorいつき)
#顔以外の領域をマスクしてから貼り付け(りゅうやorいつき)
#プレゼン(未定)
#タスク管理(りゅうや)
#目を開いてる判定だけでいいのか検討．他の処理ができるならいいけど．．．
#貼り付け先の顔画像のアングルを算出
#warpAffineを施してからもとのサイズに戻すと小さくなるので，角度から比率を算出してリサイズ
#回転のタイミングを変える


import os,sys
import cv2
import dlib
import numpy as np
import math
import datetime
from imutils import face_utils
from scipy.spatial import distance
from PIL import Image
import matplotlib.pyplot as plt

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))#これで60fps実現できるらしい
cap.set(cv2.CAP_PROP_FPS, 60)

face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt2.xml')
face_parts_detector = dlib.shape_predictor('haarcascades/shape_predictor_68_face_landmarks.dat')
detector = dlib.get_frontal_face_detector()

def calc_ear(eye):#目の検出
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    eye_ear = (A + B) / (2.0 * C)
    return round(eye_ear, 3)

def eye_marker(face_mat, position):#目の位置推定
    for i, ((x, y)) in enumerate(position):
        cv2.circle(face_mat, (x, y), 1, (255, 255, 255), -1)
        cv2.putText(face_mat, str(i), (x + 2, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

def fitting_rotated_image(img, angle):#顔の角度を回転
    height,width = img.shape[:2]
    center = (int(width/2), int(height/2))
    radians = np.deg2rad(angle)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    new_width = int(abs(np.sin(radians)*height) + abs(np.cos(radians)*width))
    new_height = int(abs(np.sin(radians)*width) + abs(np.cos(radians)*height))
    M[0,2] += int((new_width-width)/2)
    M[1,2] += int((new_height-height)/2)
    return cv2.warpAffine(img, M, (new_width, new_height))

def cvpaste(img, imgback, x, y, angle, scale):
    # x and y are the distance from the center of the background image
    r = img.shape[0]
    c = img.shape[1]
    rb = imgback.shape[0]
    cb = imgback.shape[1]
    hrb=round(rb/2)
    hcb=round(cb/2)
    hr=round(r/2)
    hc=round(c/2)
    # Copy the forward image and move to the center of the background image
    imgrot = np.zeros((rb,cb,3),np.uint8)
    imgrot[hrb-hr:hrb+hr,hcb-hc:hcb+hc,:] = img[:hr*2,:hc*2,:]
    # Rotation and scaling
    M = cv2.getRotationMatrix2D((hcb,hrb),angle,scale)
    imgrot = cv2.warpAffine(imgrot,M,(cb,rb))
    # Translation
    M = np.float32([[1,0,x],[0,1,y]])
    imgrot = cv2.warpAffine(imgrot,M,(cb,rb))
    # Makeing mask
    imggray = cv2.cvtColor(imgrot,cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(imggray, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    # Now black-out the area of the forward image in the background image
    img1_bg = cv2.bitwise_and(imgback,imgback,mask = mask_inv)
    # Take only region of the forward image.
    img2_fg = cv2.bitwise_and(imgrot,imgrot,mask = mask)
    # Paste the forward image on the background image
    imgpaste = cv2.add(img1_bg,img2_fg)
    return imgpaste

def remove_bg(
    path,
    BLUR=21,
    CANNY_THRESH_1=10,
    CANNY_THRESH_2=200,
    MASK_DILATE_ITER=10,
    MASK_ERODE_ITER=10,
    MASK_COLOR=(0.0, 0.0, 1.0),
):
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray, CANNY_THRESH_1, CANNY_THRESH_2)
    edges = cv2.dilate(edges, None)
    edges = cv2.erode(edges, None)

    contour_info = []
    _, contours, _ = cv2.findContours(
        edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    for c in contours:
        contour_info.append((
            c,
            cv2.isContourConvex(c),
            cv2.contourArea(c),
        ))
    contour_info = sorted(contour_info, key=lambda c: c[2], reverse=True)
    max_contour = contour_info[0]

    mask = np.zeros(edges.shape)
    cv2.fillConvexPoly(mask, max_contour[0], (255))

    mask = cv2.dilate(mask, None, iterations=MASK_DILATE_ITER)
    mask = cv2.erode(mask, None, iterations=MASK_ERODE_ITER)
    mask = cv2.GaussianBlur(mask, (BLUR, BLUR), 0)
    # Create 3-channel alpha mask
    mask_stack = np.dstack([mask]*3)
    # Use float matrices,
    mask_stack = mask_stack.astype('float32') / 255.0
    # for easy Blend
    img = img.astype('float32') / 255.0
    # Blend
    masked = (mask_stack * img) + ((1-mask_stack) * MASK_COLOR)
    # Convert back to 8-bit
    masked = (masked * 255).astype('uint8')

    # split image into channels
    c_red, c_green, c_blue = cv2.split(img)

    # merge with mask got on one of a previous steps
    img_a = cv2.merge((c_red, c_green, c_blue, mask.astype('float32') / 255.0))

    # show on screen (optional in jupiter)
    # plt.imshow(img_a)
    # plt.show()

    # save to disk
    cv2.imwrite('remove_background.png', img_a*255)


#ここからメイン
eyes_list = []
eyes_average_list = []#フレーム数と対応#最も目を開いている人が多いフレームを選定
faces_list = [] #顔のlistの配列[フレーム数，id(仮),目の開き具合,x,y座標,横,縦]
face_image_most_included = []

for count in range(30):#TODO 撮影ボタンを押してからフレーム左フレーム分連続撮影

    ret, rgb = cap.read()
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.11, minNeighbors=3, minSize=(100, 100))

    # ”a”を押したおときに処理
    if len(faces) > 0: #顔があると判断されたとき
        frame_faces_list = [] #このフレーム内の目の開き具合を格納するため

        for i  in range(int(len(faces))):#顔の数だけ
            x, y, w, h = faces[i, :]#顔認識の結果を格納
            # cv2.rectangle(rgb, (x, y), (x + w, y + h), (255, 0, 0), 2)# TODO:そのうち消す
            face_gray = gray[y :(y + h), x :(x + w)]
            scale = 480 / h
            face_gray_resized = cv2.resize(face_gray, dsize=None, fx=scale, fy=scale)

            face = dlib.rectangle(0, 0, face_gray_resized.shape[1], face_gray_resized.shape[0])
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

            faces_list.append([count,i,(left_eye_ear + right_eye_ear),x,y,w,h,])#顔の配列を格納[フレーム数，id(仮),目の開き具合,x,y座標,横,縦]
            frame_faces_list.append((left_eye_ear + right_eye_ear))#↑と一緒にできるはずやけどあほやから別の配列で書く

            #認識した顔を切りとる
            cropped_im = rgb[y:y+h, x:x+w]
            # 回転用に顔だけ切り出す
            face_im = rgb[y-h:y+2*h, x-w:x+2*w]#２倍で切り取り

            # 顔の角度を修正

            try:
                x_diff = face_parts[45][0] - face_parts[36][0]
                y_diff = face_parts[45][1] - face_parts[36][1]
                angle = math.degrees(math.atan2(y_diff, x_diff))
            except ZeroDivisionError as e:
                print(y_diff)
                print(x_diff)


            rotated_im = fitting_rotated_image(face_im, angle)
            img_paste_out = rotated_im.copy()
            cv2.imwrite('output_face/rotated_face_'+ str(count) +'.jpg', img_paste_out)
            # 回転後の画像で顔検出して画像保存
            cv2.imshow('kao', rotated_im)

            rotated_rects = detector(rotated_im, 1)
            if len(rotated_rects) == 0:
                print('顔が抽出されませんでした')
                continue
            rotated_rect = rotated_rects[0]
            x_start = rotated_rect.left()
            x_end = rotated_rect.right()
            y_start = rotated_rect.top()
            y_end = rotated_rect.bottom()
            cropped_im = rotated_im[y_start:y_end, x_start:x_end]
            img_paste_out = cropped_im.copy()
            cv2.imwrite('output_face/cut_face_'+ str(count)+'_'+str(i)+'.jpg', img_paste_out)
            #ここまでの内容をきれいに書く

            cv2.imwrite('output_face/outputface_'+ str(count)+'_'+str(i)+'.jpg', cropped_im)#フレーム数＋顔id.jpg
            cv2.waitKey(1)#一応止める．いらんかも
            cv2.imwrite('output_all/'+ str(count)+'.jpg', rgb)#全体画像
            cv2.waitKey(1)#一応止める．いらんかも

            if frame_faces_list[-1] < 0.5:#TODO消す：最新の顔をまばたき検知 #ここじゃないかも
                cv2.putText(rgb,"Sleepy eyes. Wake up!",
                    (10,180), cv2.FONT_HERSHEY_PLAIN, 3, (0,0,255), 3, 1)

        #フレーム内の顔画像から目の平均値を算出
        eyes_average_list.append(sum(frame_faces_list)/len(frame_faces_list))
    else:
        eyes_average_list.append(0.01)#フレームの中で抜けを作らんための行

    cv2.imshow('frame', rgb)

    if cv2.waitKey(1) == 27:
        break  # esc to quit

# 古いベース画像の選定方法
# eyes_average_array = np.array(eyes_average_list)#listからarrayに変換
# eyes_average_sort_index = np.argsort(eyes_average_array)#昇順にソートしたインデックスを返す 目が一番開いてる

# faces_array = np.array(faces_list, dtype=int)#listからarrayに変換
# index = np.argmax(faces_array[:,1])#目が一番空いている画像のindexを取得
# extraction_face_array = faces_array[index] 

# ジャンプ判定のベース画像選定方法
faces_array = np.array(faces_list, dtype=int)#listからarrayに変換
index = np.max(faces_array[:,1])#顔が一番含まれている数を出力
indexs = np.where((faces_array[:,1] == index) & (np.max(faces_array[:,4]) == faces_array[:,4]))# 顔が一番含まれているかつyが一番高い
print('--index--')
print(indexs)

        
img_base = cv2.imread('output_all/'+ str(faces_array[indexs,0])+'.jpg')  #画像読み取りimread(filename)
img_best = img_base.copy()
cv2.imwrite('best.jpg', img_best)#とりあえず選定されたベストの画像は書き出しておく#比較のために

#ベース画像の選定終わり
frame_faces_list = []#初期化ってせずにできるんかな．一応しとく#目の開き具合が最大のフレームの配列を格納#
for item in faces_list:
    if item[0] == faces_array[index,0]:
        frame_faces_list.append(item)

print(frame_faces_list)
img_paste_list = []#一応初期化
i = 0 #forのカウント用
for item in frame_faces_list:
    #if item[2] < 0.5:#最新の顔をまばたき検知
        #全体の顔画像配列からid別に抽出
    index = np.where(faces_array[:,1] == i)
    extraction_face_array = faces_array[index]#idごとに配列を抽出
    frame_index = np.argmax(extraction_face_array ,axis=0)#抽出した配列から目の開き具合が最大の配列番号を抽出それぞれの列の最大値が格納

    img_paste_list.append(1)
    img_paste_list[i] = cv2.imread('output_face/cut_face_'+ str(int(extraction_face_array[frame_index[2],[0]]))+'_'+ str(item[1]) +'.jpg')  #画像読み取りimread(filename)
    img_paste = img_paste_list[i].copy()
    cv2.imwrite('face_'+str(i)+'.jpg', img_paste)#とりあえず選定されたベストの顔画像は書き出しておく#比較のために
    #------------------------------------------------------------------------------------------------------------------------------------------
    #ベスト顔画像の貼り付け処理
    frame,face_id,eyes_open,x, y, w, h = (item)
    img_paste_resize = cv2.resize(img_paste_list[i],(w,h))#画像のリサイズ
    img_paste = img_paste_resize.copy()
    cv2.imwrite('out_face.jpg', img_paste)

    remove_bg(
    path='out_face.jpg',
    BLUR=21,
    CANNY_THRESH_1=5,
    CANNY_THRESH_2=100,
    MASK_DILATE_ITER=10,
    MASK_ERODE_ITER=6,
    )
    #img_paste = cvpaste(img_paste_resize, img_base, x, y, 0, 1)

    img_base[y:y+h,x:x+w] = img_paste_resize #はりつけ
    #img_base = cv2.add(img_base,img_paste_list[i])
    cv2.imwrite('out.jpg', img_base)
    i += 1 #カウントがうまく使えやんうざすぎおすぎ

cap.release()
cv2.destroyAllWindows()
