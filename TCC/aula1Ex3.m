clear all;
clc;
img = imread('lena512.bmp');

[M, N] = size(img);

imshow(img);

for i = 1: M
  for j = 1: N
    if img(i,j) >= 127
      imgBin(i,j) = 255;
    else
      imgBin(i,j) = 0;
    end
  end
end

figure(2), imshow(imgBin);
figure(1), imshow(img);
