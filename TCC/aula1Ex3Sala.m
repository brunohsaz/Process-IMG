img = imread('Lena512.bmp');

imgBin = img;

for i = 1:512
  for j = 1:512
    if imgBin(i,j) <= 127
       imgBin(i,j) = 0;
     else
       imgBin(i,j) = 255;
     end
  end
end

figure(1),imshow(img);
figure(2),imshow(imgBin);
