%% 1. Binarize
clc; clear;
I = imread('Images\Officer.png');
% imshow(I);
% imwrite(I, "cp_officer.png")

I50 = binarize(I, 50);
I100 = binarize(I, 100);
I150 = binarize(I, 150); 
I200 = binarize(I, 200);

subplot(2,2,1); imshow(I50); title('Seuil 50');
subplot(2,2,2); imshow(I100); title('Seuil 100');
subplot(2,2,3); imshow(I150); title('Seuil 150');
subplot(2,2,4); imshow(I200); title('Seuil 200');

%% 2. Addition, soustraction, inversion
clc; clear;
Street_Malte = imread('Images\A_Street_in_Malta.png');
Officer = imread('Images\Officer.png'); 
Officer = cat(3, Officer, Officer, Officer);

crop_region = [0, 0, 400, 400];
Street_Malte_cropped = imcrop(Street_Malte, crop_region);
Officer_cropped = imcrop(Officer, crop_region);

I_add = matrix_addition(Street_Malte_cropped, Officer_cropped);
I_sub = matrix_subtraction(Street_Malte_cropped, Officer_cropped);
SM_iv = video_inverse(Street_Malte);
O_iv = video_inverse(Officer);

subplot(2,2,1); imshow(I_add); title('Addition');
subplot(2,2,2); imshow(I_sub); title('Soustraction');
subplot(2,2,3); imshow(SM_iv); title('Inversion Street Malte');
subplot(2,2,4); imshow(O_iv); title('Inversion Officer');