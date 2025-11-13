img = imread('Lena512.bmp');

imgBin = img/85;



figure(1),imshow(img);
figure(2),colormap(gray), imagesc(imgBin);
