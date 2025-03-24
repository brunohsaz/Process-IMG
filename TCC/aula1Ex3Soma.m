clc;
clear all;
img = imread('Lena512.bmp'); %código para abrir imagem

imgClara = img + 100;

figure(1), imshow(img);
figure(2), imshow(imgClara);

imwrite (imgClara, 'AAA.bmp');
