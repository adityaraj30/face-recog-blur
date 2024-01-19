import face_recognition
import cv2 as cv
import numpy as np

from facial_landmarks_src import FaceLandmarks

f = FaceLandmarks()


def BLUR_f(t_r, t_l, b_l, b_r, src):
    img = src[b_l:t_l, b_r:t_r]  # extracting that rectangle around image only
    h, w, _ = img.shape

    img_c = img.copy()

    # your method to blur using masking
    land_m = f.get_facial_landmarks(img)
    convex_h = cv.convexHull(land_m)
    cv.imwrite('ss2.jpg', img_c)

    mk = np.zeros((h, w), np.uint8)
    cv.fillConvexPoly(mk, convex_h, 255)

    img_c = cv.blur(img_c, (30, 30))

    img_ex = cv.bitwise_and(img_c, img_c, mask=mk)

    back_mk = cv.bitwise_not(mk)

    bk = cv.bitwise_and(img, img, mask=back_mk)

    fin = cv.add(bk, img_ex)

    # returning only that part which we extracted in variable name (img) with blured face
    src[b_l:t_l, b_r:t_r] = fin
    return src


# Open the input movie file
input_movie = cv.VideoCapture("Vid_MV.mp4")
length = int(input_movie.get(cv.CAP_PROP_FRAME_COUNT))
framespersecond = int(input_movie.get(cv.CAP_PROP_FPS))
ret, frame = input_movie.read()
height, width, channels = frame.shape

# Create an output movie file (make sure resolution/frame rate matches input video!)
fourcc = cv.VideoWriter_fourcc(*'XVID')
output_movie = cv.VideoWriter('stormzyv.avi', fourcc, framespersecond, (width, height))

# Load some sample pictures and learn how to recognize them.
lmm_image = face_recognition.load_image_file("stormzy.png")
lmm_face_encoding = face_recognition.face_encodings(lmm_image)[0]

al_image = face_recognition.load_image_file("stormzy2.jpg")
al_face_encoding = face_recognition.face_encodings(al_image)[0]

known_faces = [
    lmm_face_encoding,
    al_face_encoding
]

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
frame_number = 0

while True:
    # Grab a single frame of video
    ret, frame = input_movie.read()
    frame_number += 1

    # Quit when the input video file ends
    if not ret:
        break

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    # Find all the faces and face encodings in the current frame of video
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    face_names = []
    for face_encoding in face_encodings:
        # See if the face is a match for the known face(s)
        match = face_recognition.compare_faces(known_faces, face_encoding, tolerance=0.50)

        # If you had more than 2 faces, you could make this logic a lot prettier
        # but I kept it simple for the demo
        name = None
        if match[0]:
            name = "Man 1"
        elif match[1]:
            name = "Man 2"

        face_names.append(name)

    # Label the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        if not name:
            # output_movie.write(frame)
            continue

        # Draw a box around the face
        # cv.rectangle(frame, (left - 20, top - 20), (right + 20, bottom + 20), (0, 0, 255), 2)

        width_n = right - left
        height_n = bottom - top
        # y1,x1 = b_l, y1+h,x1+w = t_r
        # left,bottom = x1,y1 ,   bottom = b_l, top = t_l,
        # right, top = x1+w, y1+h, left = b_r, right = t_r

        test_frame = BLUR_f(right, bottom, top, left ,frame)
        # Draw a label with a name below the face
        #cv.rectangle(frame, (left, bottom - 25), (right, bottom), (0, 0, 255), cv.FILLED)
        #font = cv.FONT_HERSHEY_DUPLEX
        #cv.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

    # cv.imshow("frame", frame)
    # Write the resulting image to the output video file
    print("Writing frame {} / {}".format(frame_number, length))
    # output_movie.write(frame)
    output_movie.write(test_frame)

input_movie.release()
cv.destroyAllWindows()
