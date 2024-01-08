import cv2 as cv
import mediapipe as mp
import numpy as np

from facial_landmarks_src import FaceLandmarks

# Load FaceLandmarks
fl = FaceLandmarks()

cap = cv.VideoCapture(1)

while True:
    ret, frame = cap.read()
    frame = cv.resize(frame, None, fx=0.5, fy=0.5)
    frame_copy = frame.copy()
    height, weight, _ = frame.shape

    # 1. Facial Landmark Detection (468 Facial Landmarks)
    landmarks = fl.get_facial_landmarks(frame)

    convexhull = cv.convexHull(landmarks)
    print(convexhull)  # Polygon Formation

    # print(landmarks)
    # for i in range(0, 468):
    #    pt = landmarks[i]
    #    cv.circle(frame, (pt[0], pt[1]), 5, (0, 0, 255))

    # cv.polylines(frame, [convexhull], True, (0,255,0), 3)

    # 2. Face Blurring
    mask = np.zeros((height, weight), np.uint8)
    # cv.polylines(mask, [convexhull], True, 255, 3)
    cv.fillConvexPoly(mask, convexhull, 255)

    # 3. Extracting Face
    frame_copy = cv.blur(frame_copy, (27, 27))
    face_extract = cv.bitwise_and(frame_copy, frame_copy, mask=mask)

    # 4. Extract Background
    background_mask = cv.bitwise_not(mask)
    background = cv.bitwise_and(frame, frame, mask=background_mask)

    # Final
    result = cv.add(background, face_extract)

    #cv.imshow("Frame", frame)
    #cv.imshow("Mask", mask)
    #cv.imshow("Face Extract", face_extract)
    #cv.imshow("Background Mask", background_mask)
    #cv.imshow("Background", background)
    #cv.imshow("FC", frame_copy)
    if frame is not None:
        cv.imshow("Final", result)
    ret, frame = cap.read()

    key = cv.waitKey(1)
    if key == 27:
        break

cap.release()
cv.destroyAllWindows()
