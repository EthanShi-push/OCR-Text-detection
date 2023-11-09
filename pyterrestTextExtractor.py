import pytesseract
from PIL import Image
import cv2 as cv

img = cv.imread("result.jpg")
imgNew = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
ret, binaryImage = cv.threshold(imgNew, 162, 255, cv.THRESH_BINARY)

cv.imwrite("result.jpg",binaryImage)

text = pytesseract.image_to_string(Image.open("result.jpg"))
print(text)