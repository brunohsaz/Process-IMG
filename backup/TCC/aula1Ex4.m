img = imread('lena512.bmp');
[M, N] = size(img);

img4 = zeros(M, N);
img4 = uint8(img4);
for i = 1: M
  for j = 1: N
    if img(i,j) <= 70
      img4(i,j) = 0;
      elseif
        img(i,j) <= 140
        img4(i,j) = 100;
        elseif
          img(i,j) <= 200
          img4(i,j) = 180;
    else
          img4(i,j) = 255;
    endif
  endfor
endfor

%img4 = img4;
imshow(img);
figure(2), imshow(img4);