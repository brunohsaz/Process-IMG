import cv2

loadAgt = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')

img = cv2.imread('Imagens/img4.jpg')
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = loadAgt.detectMultiScale(imgGray)

for(x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

print(faces)
cv2.imshow('Rostos', img)
cv2.waitKey()
