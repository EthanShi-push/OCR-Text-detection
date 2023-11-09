import cv2 as cv
import numpy as np


def showImage(img):
    # Display the image
    cv.imshow("Image", img)
    # Wait for the user to press a key
    cv.waitKey(0)

    # Close all windows
    cv.destroyAllWindows()


def imageProcessed(image):
    imgNew = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, binaryImage = cv.threshold(imgNew, 162, 255, cv.THRESH_BINARY)
    return binaryImage


def resize(img, width=None, height=None):
    dim = None
    (h, w) = img.shape[:2]
    if width is None and height is None:
        return img
    if height is not None:
        r = height / float(h)
        dim = (int(w * r), int(height))
    if width is not None:
        r = width / float(w)
        dim = (int(width), int(h * r))
    resizedImg = cv.resize(img, dim, interpolation=cv.INTER_AREA)
    return resizedImg


def findContours(img):
    img2 = cv.GaussianBlur(img.copy(),(5,5),0)
    edgeImg = cv.Canny(img2,75,200)
    contours, hierarchy = cv.findContours(edgeImg.copy(), cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    contour = sorted(contours,key=cv.contourArea,reverse=True)[0:5]
    wanted = None
    for c in contour:
        peri = cv.arcLength(c,True)
        approximate = cv.approxPolyDP(c,0.04*peri,True)
        if len(approximate)==4:
            wanted = approximate
            break
    wanted = wanted.reshape(-1,2)
    # img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    # cv.drawContours(img, [wanted], -1, (0, 255, 0), 2)
    # showImage(img)
    #cv.drawContours(img, contour, -1, (0,255,0), 2)

    # sort points
    #top left, top right, bottom right, bottom left
    newWanted = np.zeros((4,2),dtype="float32")

    top_left_element = wanted[np.argmin(np.sum(wanted, axis=1))]
    newWanted[0] = top_left_element

    bottom_right_element = wanted[np.argmax(np.sum(wanted, axis=1))]
    newWanted[2] = bottom_right_element
    new = wanted[
        (wanted != top_left_element).all(axis=1) &
        (wanted != bottom_right_element).all(axis=1)
        ]

    # Identify the top-right and bottom-left corners among the remaining two points
    point1, point2 = new
    if point1[0] > point2[0]:
        top_right_element = point1
        bottom_left_element = point2
    else:
        bottom_left_element = point1
        top_right_element = point2
    newWanted[1] = top_right_element
    newWanted[3] = bottom_left_element
    # 4 points of the outermost contour
    return newWanted

def transform(img,contourPts):
    (tl,tr,br,bl) = contourPts
    topWidth = np.sqrt((tr[0]-tl[0])**2 + (tr[1]-tl[1])**2)
    bottomWidth = np.sqrt((br[0]-bl[0])**2 + (br[1]-bl[1])**2)
    maxWidth = max(int(topWidth),int(bottomWidth))

    leftHeight = np.sqrt((bl[0]-tl[0])**2 + (bl[1]-tl[1])**2)
    rightHeight = np.sqrt((br[0]-tr[0])**2 + (br[1]-tr[1])**2)
    maxHeight = max(int(leftHeight),int(rightHeight))

    desiredPoints = np.array([[0,0],[maxWidth,0],[maxWidth,maxHeight],[0,maxHeight]],dtype="float32")

    matrix = cv.getPerspectiveTransform(contourPts,desiredPoints)
    desiredImg = cv.warpPerspective(img,matrix,(maxWidth,maxHeight))
    showImage(desiredImg)

def ocrTextExtracter():
    image = cv.imread("test.jpg")
    assert image is not None, "No such file"

    processedImg = imageProcessed(image)
    ratio = processedImg.shape[0] / 500.0
    copy_img = processedImg.copy()

    resizedImg = resize(processedImg, height=500)

    outerContour = findContours(resizedImg)

    transformImg = transform(image,outerContour.reshape(4,2)*ratio)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    ocrTextExtracter()
