import cv2
import os

image = cv2.imread("images_data/img97.jpg")
hsvImage = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

cv2.imwrite("test_image/hsvimage.jpg", hsvImage)
