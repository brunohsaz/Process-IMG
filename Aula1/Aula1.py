import cv2

loadAgt = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')

img = cv2.imread('Imagens/img4.jpg')
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = loadAgt.detectMultiScale(imgGray, scaleFactor=1.05, minNeighbors=2, minSize=(20, 20))

print(faces)

for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

cv2.imshow('Faces', img)
cv2.waitKey()