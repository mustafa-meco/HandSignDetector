import time

import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import os


cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

offset = 20
imgSize = 300


counter = 0

alphabetlist = ['1. Alef', '2. Baa', '3. Taa', '4. Thaa', '5. Geem', '6. Haa', '7. Khaa', '8. Dal', '9. Zaal', '10. Raa', '11. Zeen', '12. Seen', '13. Sheen', '14. Saad', '15. Daad', '16. Tah', '17. Zah', '18. Aen', '19. ghen', '20. Faa', '21. Kaaf', '22. Kaf', '23. Laam', '24. Meem', '25. Noon', '26. Heh', '27. Wow', '28. LaamAlf', '29. Yaa']
wordCount = -1

while True:
    success, img = cap.read()

    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x,y,w,h = hand['bbox']

        imgWhite = np.ones((imgSize,imgSize,3),np.uint8) * 255

        imgCrop = img[y-offset:y+h+offset,x-offset:x+w+offset]

        imgCropShape = imgCrop.shape


        aspectRatio = h/w

        if aspectRatio > 1:
            k = imgSize/h
            wCal = math.ceil(k*w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape

            wGap = math.ceil((imgSize - wCal)/2)

            imgWhite[:, wGap:wCal+wGap] = imgResize

        else:
            k = imgSize/w
            hCal = math.ceil(k*h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape

            hGap = math.ceil((imgSize - hCal)/2)

            imgWhite[hGap:hCal+hGap, :] = imgResize



        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)

    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key == ord("q"):
        break
    if key == ord("s"):
        folder = f'Data/{alphabetlist[wordCount]}'
        counter += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg',imgWhite)
        print(counter)
    if key==ord("i"):
        counter = 0
        wordCount += 1
        print(alphabetlist[wordCount])