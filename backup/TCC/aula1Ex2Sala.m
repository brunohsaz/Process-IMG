clc;
clear all;
img = imread('Lena512.bmp'); %código para abrir imagem

for i = 1: 512
  for j = 1: 512
    imgClara(i,j) = img(i,j) + 100;
  endfor
endfor

figure(1), imshow(img);
figure(2), imshow(imgClara);
