
# Importing all the required modules
import cv2 as cv
import mediapipe as mp
import time
import math

# Adopting the approach of object oriented programming
# Defining a class
# Many different methods added in it
class hand_detector():
    def __init__(self, mode = False, maxHand = 2, detectionCon = 0.5, trackCon = 0.5):

        # Creating variables of an object
        self.mode = mode
        self.maxHand = maxHand
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.mphands = mp.solutions.hands
        self.hands = self.mphands.Hands(self.mode, self.maxHand, self.detectionCon, self.trackCon)
        self.mpdraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]

    # The below method helps to read information of hands detected and connects all the points detected on the hand
    def findHands(self, image, draw = True):
        imageRGB = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        self.results = self.hands.process(imageRGB)

        if self.results.multi_hand_landmarks:
            for handlms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpdraw.draw_landmarks(image, handlms, self.mphands.HAND_CONNECTIONS) # used this to draw the 21 point and connect them via lines.

        return image
    # Defining a function to get the coordinates of each landmark point.
    def findPosition(self, image, handNo = 0, draw = True):

        xList = []
        yList = []
        bbox = []

        self.lmList = []
        if self.results.multi_hand_landmarks:
            my_hand = self.results.multi_hand_landmarks[handNo]


            for id, land_mark in enumerate(my_hand.landmark): # id is for each corresponding point, land_mark give the x,y,z coordinates of each point
                #print(id, land_mark)
                height, width, channel = image.shape
                cx, cy = int(land_mark.x*width) , int(land_mark.y * height) # here we have converted the x,y coordinates to pixel coordinates.
                #print(id, cx, cy)   
                xList.append(cx)
                yList.append(cy)
                self.lmList.append([id,cx,cy])
                if draw:

                
                    cv.circle(image, (cx,cy), 3, (255,0,255), cv.FILLED)
            xmin, xmax = min(xList), max(xList)
            ymin, ymax = min(yList), max(yList)
            bbox = xmin, ymin, xmax, ymax

            if draw:
                cv.rectangle(image, (xmin - 20, ymin - 20), (xmax + 20, ymax + 20),
                              (0, 255, 0), 2)

        return self.lmList, bbox
    # Defining a function to identify the status of fingers (up or down) and assigning then to input a particular command.
    def fingersUp(self):
        fingers = []
        # Thumb
        if len(self.lmList) != 0:
            if self.lmList[self.tipIds[0]][1] > self.lmList[self.tipIds[0] - 1][1]:
                fingers.append(1)
            else:
                fingers.append(0)

        # Fingers
            for id in range(1, 5):

                if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 2][2]:
                    fingers.append(1)
                else:
                    fingers.append(0)

        # totalFingers = fingers.count(1)

        return fingers
    # Defining a function to find the distance between the required landmark points.
    def findDistance(self, p1, p2, image, draw=True,r=15, t=3):
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if draw:
            cv.line(image, (x1, y1), (x2, y2), (255, 0, 255), t)
            cv.circle(image, (x1, y1), r, (255, 0, 255), cv.FILLED)
            cv.circle(image, (x2, y2), r, (255, 0, 255), cv.FILLED)
            cv.circle(image, (cx, cy), r, (0, 0, 255), cv.FILLED)
        length = math.hypot(x2 - x1, y2 - y1)

        return length, image, [x1, y1, x2, y2, cx, cy]     

def main():
    # Capturing video via webcam
    cap = cv.VideoCapture(0)
    previous_time = 0
    current_time = 0
    detector = hand_detector()

    while True:
        # Reading the images through webcam
        success, image = cap.read()

        # Finding the hand landmark
        # bbox = bounding box around the detected hand
        image = detector.findHands(image)
        lmList, bbox = detector.findPosition(image)
        if len(lmList) != 0:
            print(lmList[4])


        current_time = time.time()
        fps = 1 / (current_time - previous_time)
        previous_time = current_time

        # To display frame rate on the displaying window
        cv.putText(image, str(int(fps)), (10, 70), cv.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 3)

        # Display window
        cv.imshow('image', image)
        cv.waitKey(1)

if __name__ == "__main__":
    main()
