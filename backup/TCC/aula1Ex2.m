clear all;
clc;
img = imread('mulher.jpg');

for i = 1: 512
  for j = 1: 512
    imgClara(i,j) = img(i,j) + 50;
  endfor
endfor
figure(1), imshow(img);
figure(2), imshow(imgClara);

