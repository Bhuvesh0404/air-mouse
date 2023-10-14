
# Importing all the required modules
import cv2 as cv
import Hand_Tracking as htm
import numpy as np
import time
import pyautogui

# Height and Width of cam respectively
wCam = 640 
hCam = 480

# Capturing video and getting dimensions of window
cap = cv.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

# Parameters that would be used in frame rate further
# Previous time and Current time respectively
pTime = 0
cTime = 0

# Creating an object detector and storing the class in it, defined in Hand_Tracking module
detector = htm.hand_detector(maxHand = 1)

# Getting Width and Height of the screen
wScr, hScr = pyautogui.size()

# Frame Reduction
frameR = 100

# Initializing smoothening values. It will be useful further in code
smooth = 5

# Parameters used in further smoothening of values
# Previous location of X and Y
plocX, plocY = 0, 0

# Current location of X and Y
clocX, clocY = 0, 0


while True:
    # Reading the images through webcam
    success, img = cap.read()

    # Finding the hand landmark
    # bbox = bounding box around the detected hand
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img)

    # Getting coordinates of tip of index and middle finger respectively and printing them in the terminal window
    if len(lmList) != 0:
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]

        print(x1, y1, x2, y2)

    # Checking which fingers are up
    fingers = detector.fingersUp()
    print(fingers)

    # Scaling down the actual screen to the rectangle of below mentioned dimensions in order cover whole screen through the rectangle
    cv.rectangle(img, (frameR, frameR), (wCam-frameR, hCam - frameR), (255, 0, 255), thickness = 3)

    # If only index finger is up, then mouse in moving mode
    if len(fingers) != 0:
        if fingers[1] == 1 and fingers[2] == 0 and fingers[0] == 0 and fingers[3] == 0 and fingers[4] == 0:

            # Converting Coordinates in order to make a proper sync between actual coordinates and that on screen
            x3 = np.interp(x1, (frameR, wCam-frameR), (0, wScr))
            y3 = np.interp(y1, (frameR, hCam-frameR), (0, hScr))

            # Smoothening values
            clocX = plocX + (x3-plocX) / smooth
            clocY = plocY + (y3-plocY) / smooth

            # Moving mouse by keeping only the index finger up moving it
            pyautogui.moveTo(wScr - clocX, clocY, duration = 0.005)

            # Shows a big circle at the tip of the index finger when in moving mode
            cv.circle(img, (x1, y1), 10, (255, 0, 255), cv.FILLED)
            plocX, plocY = clocX, clocY

        # In left clicking mode when both the fingers (index and middle fingers) are up
        if fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 0 and fingers[4] == 0 and fingers[0] == 0:

            # Finding distance between index and middle finger
            length, img, lineInfo = detector.findDistance(8, 12, img)
            print(length)

            # Left Clicking command when distance between index and middle finger is less than certain value
            # A circular dot of fixed radius and colour appears when in the gesture of clicking command
            if length < 35:
                cv.circle(img, (lineInfo[4], lineInfo[5]), 10, (0, 255, 0), cv.FILLED)
                pyautogui.click()

        # In right clicking mode when both the fingers (index finger and thumb) are up
        if fingers[1] == 1 and fingers[2] == 0 and fingers[3] == 0 and fingers[4] == 0 and fingers[0] == 1:

            # Finding distance between index finger and thumb
            length, img, lineInfo = detector.findDistance(8, 4, img)
            print(length)

            # Right Clicking command when distance between index finger and thumb is less than certain value
            # A circular dot of fixed radius and colour appears when in the gesture of clicking command
            if length < 35:
                 cv.circle(img, (lineInfo[4], lineInfo[5]), 10, (0, 255, 0), cv.FILLED)
                 pyautogui.click(button='right')

        # In Scroll up mode when index and smallest fingers are up keeping rest all fingers down
        if fingers[4] == 1 and fingers[1] == 1 and fingers[0] == 0 and fingers[2] == 0 and fingers[3] == 0:
            pyautogui.scroll(200)

        # In Scroll down mode when only smallest finger is up keeping rest all fingers down
        if fingers[4] == 1 and fingers[1] == 0 and fingers[0] == 0 and fingers[2] == 0 and fingers[3] == 0:
            pyautogui.scroll(-200)

        # Press space bar when only thumb is up keeping rest all fingers down
        if fingers[0] == 1 and fingers[1] == 0 and fingers[2] == 0 and fingers[3] == 0 and fingers[4] == 0:
            pyautogui.typewrite(['space'])

    # Frame Rate
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    # To display frame rate on the displaying window
    cv.putText(img, str(int(fps)), (10, 50),  cv.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 3)


    # Display window
    cv.imshow('img', img)
    cv.waitKey(1)
   



