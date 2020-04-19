import numpy as np
import cv2
import pickle


def gstreamer_pipeline(
        capture_width=3280,
        capture_height=2464,
        display_width=820,
        display_height=616,
        framerate=21,
        flip_method=0,
):
    return (
            "nvarguscamerasrc ! "
            "video/x-raw(memory:NVMM), "
            "width=(int)%d, height=(int)%d, "
            "format=(string)NV12, framerate=(fraction)%d/1 ! "
            "nvvidconv flip-method=%d ! "
            "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
            "videoconvert ! "
            "video/x-raw, format=(string)BGR ! appsink"
            % (
                capture_width,
                capture_height,
                framerate,
                flip_method,
                display_width,
                display_height,
            )
    )


def face_recognize():
    face_cascade = cv2.CascadeClassifier(
        "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml"
    )

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("trainner.yml")
    labels = {"person_name": 1}
    with open("labels.pickle", 'rb') as f:
        og_labels = pickle.load(f)
        labels = {v: k for k, v in og_labels.items()}

    cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
    if cap.isOpened():
        cv2.namedWindow("Face Recognize", cv2.WINDOW_AUTOSIZE)
        while cv2.getWindowProperty("Face Recognize", 0) >= 0:
            ret, img = cap.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                roi_gray = gray[y: y + h, x: x + w]
                roi_color = img[y: y + h, x: x + w]

                id_, conf = recognizer.predict(roi_gray)
                if conf >= 45 and conf <= 85:
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    name = labels[id_]
                    color = (255, 255, 255)
                    stroke = 2
                    cv2.putText(img, name, (x, y), font, 1, color, stroke, cv2.LINE_AA)

                color = (0, 0, 0)
                stroke = 2
                end_cord_x = x + w
                end_cord_y = y + h
                cv2.rectangle(img, (x, y), (end_cord_x, end_cord_y), color, stroke)

            cv2.imshow("Face Recognize", img)
            keyCode = cv2.waitKey(30) & 0xFF
            # Stop the program on the ESC key
            if keyCode == 27:
                break

        cap.release()
        cv2.destroyAllWindows()
    else:
        print("Unable to open camera")


if __name__ == "__main__":
    face_recognize()
