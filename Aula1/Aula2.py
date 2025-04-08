import cv2
dir = "/home/bruno/Documentos/Process-IMG/Aula1/"

loadAgtFace = cv2.CascadeClassifier(dir+'haarcascades/haarcascade_frontalface_default.xml')
loadAgtEye = cv2.CascadeClassifier(dir+'haarcascades/haarcascade_eye.xml')

img = cv2.imread(dir+'imagens/img5.jpg')
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = loadAgtFace.detectMultiScale(imgGray)

print(faces)

for (x, y, w, h) in faces:
    img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    olho = img[y:y+h, x:x+w]
    olhoGray = cv2.cvtColor(olho, cv2.COLOR_BGR2GRAY)
    olhoDetec = loadAgtEye.detectMultiScale(olhoGray)

    for(ox, oy, ow, oh) in olhoDetec:
        cv2.rectangle(olho, (ox, oy), (ox + ow, oy + oh), (0, 255, 0), 2)

cv2.imshow('Faces', img)
cv2.waitKey()