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
import numpy
import math
import datetime
from imutils import face_utils
from scipy.spatial import distance
from PIL import Image
import glob

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))#これで60fps実現できるらしい
cap.set(cv2.CAP_PROP_FPS, 60)

face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt2.xml')
face_parts_detector = dlib.shape_predictor('haarcascades/shape_predictor_68_face_landmarks.dat')
detector = dlib.get_frontal_face_detector()


class NoFaces(Exception):
    pass

class Face:
  def __init__(self, image, rect):
    self.image = image
    self.landmarks = numpy.matrix([[p.x, p.y] for p in face_parts_detector(image, rect).parts()])

class BeBean:
  SCALE_FACTOR = 1
  FEATHER_AMOUNT = 11

  # 特徴点のうちそれぞれの部位を表している配列のインデックス
  FACE_POINTS = list(range(17, 68))
  MOUTH_POINTS = list(range(48, 61))
  RIGHT_BROW_POINTS = list(range(17, 22))
  LEFT_BROW_POINTS = list(range(22, 27))
  RIGHT_EYE_POINTS = list(range(36, 42))
  LEFT_EYE_POINTS = list(range(42, 48))
  NOSE_POINTS = list(range(27, 35))
  JAW_POINTS = list(range(0, 17))

  ALIGN_POINTS = (LEFT_BROW_POINTS + RIGHT_EYE_POINTS + LEFT_EYE_POINTS + RIGHT_BROW_POINTS +
    NOSE_POINTS + MOUTH_POINTS)

  # オーバーレイする特徴点
  OVERLAY_POINTS = [LEFT_EYE_POINTS + RIGHT_EYE_POINTS + LEFT_BROW_POINTS + RIGHT_BROW_POINTS,
    NOSE_POINTS + MOUTH_POINTS]

  COLOR_CORRECT_BLUR_FRAC = 0.7

  def __init__(self, before_after = True):
    self.detector = dlib.get_frontal_face_detector()
    self._load_beans()
    self.before_after = before_after

  def load_faces_from_image(self, image_path):
    """
      画像パスから画像オブジェクトとその画像から抽出した特徴点を読み込む。
      ※ 画像内に顔が1つないし複数検出された場合も、返すので正確には「特徴点配列」の配列を返す
    """
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (image.shape[1] * self.SCALE_FACTOR,
                               image.shape[0] * self.SCALE_FACTOR))

    rects = self.detector(image, 1)

    if len(rects) == 0:
      raise NoFaces
    else:
      print("Number of faces detected: {}".format(len(rects)))

    faces = [Face(image, rect) for rect in rects]
    return image, faces

  def transformation_from_points(self, t_points, o_points):
    """
      特徴点から回転やスケールを調整する。
      t_points: (target points) 対象の特徴点(入力画像)
      o_points: (origin points) 合成元の特徴点(つまりビーン)
    """

    t_points = t_points.astype(numpy.float64)
    o_points = o_points.astype(numpy.float64)

    t_mean = numpy.mean(t_points, axis = 0)
    o_mean = numpy.mean(o_points, axis = 0)

    t_points -= t_mean
    o_points -= o_mean

    t_std = numpy.std(t_points)
    o_std = numpy.std(o_points)

    t_points -= t_std
    o_points -= o_std

    # 行列を特異分解しているらしい
    # https://qiita.com/kyoro1/items/4df11e933e737703d549
    U, S, Vt = numpy.linalg.svd(t_points.T * o_points)
    R = (U * Vt).T

    return numpy.vstack(
      [numpy.hstack((( o_std / t_std ) * R, o_mean.T - ( o_std / t_std ) * R * t_mean.T )),
      numpy.matrix([ 0., 0., 1. ])]
    )

  def get_face_mask(self, face):
    image = numpy.zeros(face.image.shape[:2], dtype = numpy.float64)
    for group in self.OVERLAY_POINTS:
      self._draw_convex_hull(image, face.landmarks[group], color = 1)

    image = numpy.array([ image, image, image ]).transpose((1, 2, 0))
    image = (cv2.GaussianBlur(image, (self.FEATHER_AMOUNT, self.FEATHER_AMOUNT), 0) > 0) * 1.0
    image = cv2.GaussianBlur(image, (self.FEATHER_AMOUNT, self.FEATHER_AMOUNT), 0)

    return image

  def warp_image(self, image, M, dshape):
    output_image = numpy.zeros(dshape, dtype = image.dtype)
    cv2.warpAffine(
      image,
      M[:2],
      (dshape[1], dshape[0]),
      dst = output_image, borderMode = cv2.BORDER_TRANSPARENT, flags = cv2.WARP_INVERSE_MAP
    )
    return output_image

  def correct_colors(self, t_image, o_image, t_landmarks):
    """
      対象の画像に合わせて、色を補正する
    """
    blur_amount = self.COLOR_CORRECT_BLUR_FRAC * numpy.linalg.norm(
      numpy.mean(t_landmarks[self.LEFT_EYE_POINTS], axis = 0) -
      numpy.mean(t_landmarks[self.RIGHT_EYE_POINTS], axis = 0)
    )
    blur_amount = int(blur_amount)

    if blur_amount % 2 == 0: blur_amount += 1

    t_blur = cv2.GaussianBlur(t_image, (blur_amount, blur_amount), 0)
    o_blur = cv2.GaussianBlur(o_image, (blur_amount, blur_amount), 0)

    # ゼロ除算を避ける　
    o_blur += (128 * (o_blur <= 1.0)).astype(o_blur.dtype)

    return (o_image.astype(numpy.float64) * t_blur.astype(numpy.float64) / o_blur.astype(numpy.float64))

  def to_bean(self, image_path):
    original, faces = self.load_faces_from_image(image_path)

    # base_imageに合成していく
    base_image = original.copy()

    #ここから編集
    for face in faces:
      bean = self._get_bean_similar_to(face)
      bean_mask = self.get_face_mask(bean)

      M = self.transformation_from_points(
        face.landmarks[self.ALIGN_POINTS],
        bean.landmarks[self.ALIGN_POINTS]
      )

      warped_bean_mask = self.warp_image(bean_mask, M, base_image.shape)
      combined_mask = numpy.max(
        [self.get_face_mask(face), warped_bean_mask], axis = 0
      )

      warped_image = self.warp_image(bean.image, M, base_image.shape)
      warped_corrected_image = self.correct_colors(base_image, warped_image, face.landmarks)
      base_image = base_image * (1.0 - combined_mask) + warped_corrected_image * combined_mask

    path, ext = os.path.splitext( os.path.basename(image_path) )
    cv2.imwrite('outputs/output_' + path + ext, base_image)

    if self.before_after is True:
      before_after = numpy.concatenate((original, base_image), axis = 1)
      cv2.imwrite('before_after/' + path + ext, before_after)

  def _draw_convex_hull(self, image, points, color):
    "指定したイメージの領域を塗りつぶす"

    points = cv2.convexHull(points)
    cv2.fillConvexPoly(image, points, color = color)

  def _load_beans(self):
    "Mr. ビーンの画像をロードして、顔(特徴点など)を検出しておく"

    self.beans = []
    for image_path in glob.glob(os.path.join('input', '*.jpg')):#TODO:ここで似ている顔のフォルダを列挙
      image, bean_face = self.load_faces_from_image(image_path)
      self.beans.append(bean_face[0])
    print('Mr. Beanをロードしました.')

  def _get_bean_similar_to(self, face):
    "特徴点の差分距離が小さいMr.ビーンを返す"

    get_distances = numpy.vectorize(lambda bean: numpy.linalg.norm(face.landmarks - bean.landmarks))

    distances = get_distances(self.beans)
    return self.beans[distances.argmin()]


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

#ここからメイン
eyes_list = []
eyes_average_list = []#フレーム数と対応#最も目を開いている人が多いフレームを選定
faces_list = [] #顔のlistの配列[フレーム数，id(仮),目の開き具合,x,y座標,横,縦]

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

            x_diff = face_parts[45][0] - face_parts[36][0]
            y_diff = face_parts[45][1] - face_parts[36][1]
            angle = math.degrees(math.atan2(y_diff, x_diff))

            faces_list.append([count,i,(left_eye_ear + right_eye_ear),x,y,w,h,angle])#顔の配列を格納[フレーム数，id(仮),目の開き具合,x,y座標,横,縦,顔の角度]
            frame_faces_list.append((left_eye_ear + right_eye_ear))#↑と一緒にできるはずやけどあほやから別の配列で書く

            #認識した顔を切りとる
            cropped_im = rgb[y:y+h, x:x+w]

            cv2.imwrite('output_face/outputface_'+ str(count)+'_'+str(i)+'.jpg', cropped_im)#フレーム数＋顔id.jpg
            cv2.waitKey(1)#一応止める．いらんかも
            cv2.imwrite('output_all/'+ str(count)+'.jpg', rgb)#全体画像
            cv2.waitKey(1)#一応止める．いらんかも

        #フレーム内の顔画像から目の平均値を算出
        eyes_average_list.append(sum(frame_faces_list)/len(frame_faces_list))
    else:
        eyes_average_list.append(0.01)#フレームの中で抜けを作らんための行

    cv2.imshow('frame', rgb)

    if cv2.waitKey(1) == 27:
        break  # esc to quit

#ベース画像の選定
faces_array = np.array(faces_list, dtype=int)#listからarrayに変換
index_tmp = np.argmax(faces_array[:,1])#顔が一番含まれている数を出力
print('--indextemp--')
print(index_tmp)
index = np.where((faces_array[:,1] == index_tmp) & (np.min(faces_array[:,4]) == faces_array[:,4]))# 顔が一番含まれているかつyが一番高い
print('--index--')
print(index[0])
extraction_face_array = faces_array[index]#idごとに配列を抽出

img_base = cv2.imread('output_all/'+ str(faces_array[index[0][0],0])+'.jpg')  #画像読み取りimread(filename)
img_best = img_base.copy()
datetime = str(datetime.datetime.now())
cv2.imwrite('best'+datetime+'.jpg', img_best)#とりあえず選定されたベストの画像は書き出しておく#比較のために

#ベース画像の選定終わり#TODO:ジャンプの瞬間を検知する処理入れば少し書き換え

#ベストの顔画像切り取りーここから
frame_faces_list = []#初期化ってせずにできるんかな．一応しとく#目の開き具合が最大のフレームの配列を格納#
for item in faces_list:
    if item[0] == faces_array[index,0]:#ここおかしい
        frame_faces_list.append(item)

img_paste_list = []#一応初期化
i = 0 #forのカウント用
for item in frame_faces_list:
    #if item[2] < 0.5:#最新の顔をまばたき検知
        #全体の顔画像配列からid別に抽出
    index = np.where(faces_array[:,1] == i)
    extraction_face_array = faces_array[index]#idごとに配列を抽出
    frame_index = np.argmax(extraction_face_array ,axis=0)#抽出した配列から目の開き具合が最大の配列番号を抽出それぞれの列の最大値が格納

    img_paste_list.append(1)
    img_paste_list[i] = cv2.imread('output_face/outputface_'+ str(int(extraction_face_array[frame_index[2],[0]]))+'_'+ str(item[1]) +'.jpg')  #画像読み取りimread(filename)
    img_paste = img_paste_list[i].copy()
    cv2.imwrite('input/' + str(i)+'.jpg', img_paste)#とりあえず選定されたベストの顔画像は書き出しておく#比較のために
    #------------------------------------------------------------------------------------------------------------------------------------------
    #ベスト顔画像の貼り付け処理



#ベストの顔画像切り取りーここまで

#ミスター・ビーンの処理をする
be_bean = BeBean()
be_bean.to_bean('best'+datetime+'.jpg')

cap.release()
cv2.destroyAllWindows()
