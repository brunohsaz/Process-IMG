import cv2

webCam = cv2.VideoCapture(0)
classifier = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')

while True:
    cam, frame = webCam.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detect = classifier.detectMultiScale(gray)
    for(x, y, w, h) in detect:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) == ord('q'):
        break;

webCam.release()
cv2.destroyAllWindows()