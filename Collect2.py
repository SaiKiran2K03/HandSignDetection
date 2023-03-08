import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

cap = cv2.VideoCapture(0)
detector = HandDetector(detectionCon=0.8, maxHands=2)
offset = 125
offset1= 20
imgSize = 300
folder = "Data/3"
counter = 0

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)

    if hands:

        hand1 = hands[0]
        lmList1 = hand1["lmList"]
        x,y,w,h = hand1["bbox"]
        cx,cy = hand1["center"]
        centerPoint1 = cx,cy
        handType1 = hand1["type"]
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = img[y - offset1:cy + h + offset1, x - offset1:cx + w + offset1]
        imgCropShape = imgCrop.shape


        fingers1 = detector.fingersUp(hand1)

        aspectRatio = h / w
        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize

        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize


        if len(hands) == 2:
            hand2 = hands[1]
            lmList2 = hand2["lmList"]
            x,y,w,h = hand2["bbox"]
            cx1,cy1 = hand2["center"]
            centerPoint2 = cx1,cy1
            px=(cx1+cx)//2
            py=(cy1+cy)//2
            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            imgCrop = img[py - offset:py + offset, x - offset:px+ offset]

            imgCropShape = imgCrop.shape

            handType2 = hand2["type"]

            fingers2 = detector.fingersUp(hand2)

            length, info, img = detector.findDistance(centerPoint1, centerPoint2, img)

            aspectRatio = h / w
            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                imgResizeShape = imgResize.shape
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap:wCal + wGap] = imgResize

            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                imgResizeShape = imgResize.shape
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgResize
        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)

    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key == ord(" ") and counter <3000:
        counter += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg',imgWhite)
        # print(f'{folder}/Image_{time.time()}.jpg', imgWhite)
        print(counter)
