import cv2 as cv
import mediapipe as mp

# Loading Face Mash
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

# Image
image = cv.imread('person.jpg')
height, width, _ = image.shape
#print(height, width)

# Mediapipe processes the image in RGB format compared to other libraries (like OpenCV) which processes the image in a
# BGR format, thus need to convert.

rgb_image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

#Facial Landmarks
result = face_mesh.process(rgb_image)

for facial_landmarks in result.multi_face_landmarks:
    for i in range(0,468):
        pt1 = facial_landmarks.landmark[i]
        x = int(pt1.x * width)
        y = int(pt1.y * height)

        cv.circle(image, (x,y), 1, (200,100,0), -1)
        #cv.putText(image, str(i), (x,y), 0, 0.1, (0,0,0))


#cv.imshow('RGB_Image', rgb_image)
cv.imshow('Image', image)
cv.waitKey(0)
