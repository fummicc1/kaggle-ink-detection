import cv2
import matplotlib.pyplot as plt
import os

image_path = input("image path: ")

if not os.path.exists(image_path):
    image_path = os.path.join(os.getcwd(), image_path)

# 画像の読み込み 
img = cv2.imread(image_path)

# カラーデータの色空間の変換 
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# 画像の保存
png_image_path = image_path.split(".")[0] + ".png"
cv2.imwrite(png_image_path, img)
