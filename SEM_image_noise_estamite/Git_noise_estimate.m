clear
clc

imageblock_path = './image_block.tif';
imageblock = imread(imageblock_path);

imageblock_row = 5;
imageblock_col = 5;
imageblock_num = imageblock_row*imageblock_row;

img_avg = zeros(1,imageblock_num);
img_std = zeros(1,imageblock_num);
img_var = zeros(1,imageblock_num);

countnum = 1;
for i = 0:4
    for j = 0:4
        img = imageblock(i * 100 + 1:(i + 1) * 100, j * 100 + 1:(j + 1) * 100);
        img = double(img);
        img_avg(countnum) = mean2(img);
        img_std(countnum) = std2(img);
        img_var(countnum) = img_std(countnum) * img_std(countnum);
        countnum = countnum + 1;
    end
end

%% Forward analysis 
x = linspace(1,imageblock_num,imageblock_num);
[img_avgnew,img_avgind]=sort(img_avg,'ascend'); 
img_stdnew = img_std(img_avgind);
img_varnew = img_var(img_avgind);

figure
line1 = polyfit(img_avgnew,img_varnew,1);
min_avg = min(img_avgnew);
max_avg = max(img_avgnew);
t_g2 = min_avg - 0.1:1e-06:max_avg + 0.1;
p_g2 = polyval(line1,t_g2);
plot(img_avgnew,img_varnew,'o',t_g2,p_g2);
% legend('ÄâºÏµÄÇúÏß');
title('The relationship analysis between the image blocks'' means and variances');
xlabel('The means of the image blocks');
ylabel('The variances of the image blocks');

figure
img_avgnew = img_avgnew / 10;
plot(x,img_avgnew,'g',x,img_varnew,'b--o')
title('The relevance analysis of the image blocks'' means and variances');
xlabel('Index number of the image sorted by pixel intensity (ascending order)'); 
ylabel('The means / variances of the image blocks');

%% Reverse analysis(QQ plot)
figure
rowtemp = 0;
coltemp = 3;
imgtemp = imageblock(rowtemp * 100 + 1:(rowtemp + 1) * 100, coltemp * 100 + 1:(coltemp + 1) * 100);
Lambda = mean2(imgtemp);
[m,n] = size(imgtemp);
atemp = 1 / line1(1);
imtempnoise = random('Poisson',Lambda*atemp,m,n);
imtempnoise = imtempnoise / atemp;
imtempnoise = max(1, imtempnoise);
imtempnoise = min(imtempnoise, 256);

if line1(2) > 0
    Gaussian_noise = random('Normal',0,line1(2),m,n);
    imtempnoise=imtempnoise+Gaussian_noise;
else
    Gaussian_noise = random('Normal',0,-line1(2),m,n);
    imtempnoise=imtempnoise-Gaussian_noise;
end
qqplot(double(imgtemp(:)),imtempnoise(:));
hold on
