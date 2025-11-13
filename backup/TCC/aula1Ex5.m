img = imread('Lena512.bmp');

img8 = img/32;
img8 = img8*32;

figure(1), imshow(img);
figure(2), imshow(img8);
