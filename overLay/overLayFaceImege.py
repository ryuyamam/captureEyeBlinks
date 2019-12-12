import cv2
import numpy as np
from PIL import Image


class overLayClass():
    def __init__(self):
        print("init ovaerLayClass")
        # 初期化したい変数があればここで

    # 画像のオーバーレイ処理を行うクラス
    # srcの引数画像に対して、固定の画像をオーバーレイしreturnする
    # @param src:撮影画像 location:x,y座標
    # @return オーバーレイ後の画像
    def overLayImage(self, src, location):
        overlay = cv2.imread("overLay/overLayImage/dammy.png", cv2.IMREAD_UNCHANGED)
        overlay_height, overlay_width = overlay.shape[:2]

        # 背景をPIL形式に変換
        src = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
        pil_src = Image.fromarray(src)
        pil_src = pil_src.convert('RGBA')

        # オーバーレイをPIL形式に変換
        overlay = cv2.cvtColor(overlay, cv2.COLOR_BGRA2RGBA)
        pil_overlay = Image.fromarray(overlay)
        pil_overlay = pil_overlay.convert('RGBA')

        # 画像を合成
        pil_tmp = Image.new('RGBA', pil_src.size, (255, 255, 255, 0))
        pil_tmp.paste(pil_overlay, location, pil_overlay)
        result_image = Image.alpha_composite(pil_src, pil_tmp)

        # OpenCV形式に変換
        return cv2.cvtColor(np.asarray(result_image), cv2.COLOR_RGBA2BGRA)
