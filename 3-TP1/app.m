%% I.1. Binarize
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

%% I.2. Addition, soustraction, inversion
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

figure;
subplot(2,2,1); imshow(I_add); title('Addition');
subplot(2,2,2); imshow(I_sub); title('Soustraction');
subplot(2,2,3); imshow(SM_iv); title('Inversion Street Malte');
subplot(2,2,4); imshow(O_iv); title('Inversion Officer');

%% II.1. Lecture, affichage, écriture d'une image
clc; clear;

I = imread('Images\Helicoptere.ppm');
imshow(I);
imwrite(I, 'out/my-img.ppm');

% Q1: L'image I est représentée par 3 matrices de 427x805 pixels: une
% matrices pour les composantes rouge, une pour les composantes vertes et
% une pour les composantes bleues de l'image.

%Ing = rgb2gray(I);
% ou
Ing = (I(:,:,1) + I(:,:,2) + I(:,:,3))/3;

imshow(Ing);

% Parmi les formats d'images supportés sur Matlab, on trouve les fichier
% PNG, JPEG ou encore TIF.

%% II.2. Egalisation, réhaussement, recadrage d'histogramme

I1 = imadjust(Ing);
I2 = histeq(Ing);

figure;
subplot(3,2,1), imshow(Ing), title('Ing'), subplot(3,2,2), imhist(Ing);
subplot(3,2,3), imshow(I1), title('imadjust'), subplot(3,2,4), imhist(I1);
subplot(3,2,5), imshow(I2), title('histeq'), subplot(3,2,6), imhist(I2);

% Q2: 
% - L'image initiale est relativement sombre et son histogramme est 
% concentré entre 0 et 80 d'intensité avec des barres très collées (donc bcp d'intensités proches).
% - Avec `imadjust`, l'image très éclaircie. Les intensités sont réparties
% sur tout le spectre, avec légèrement plus de valeurs dans les
% hautes intensités.
% - Avec `histeq`, l'image est dans un niveau de gris intermédiaire mais 
% est plusgranuleuse. Les intensités sont plus espacées, comprises entre 0 4
% et 120. On observe également un pic à 180 environ.


%% II.3. Filtrage linéaire médian et adaptatif

J = imnoise(Ing, 'salt & pepper', 0.08);
K = imfilter(J, fspecial('average', 5));
K1 = imfilter(J, fspecial("average", 9));
L = medfilt2(J, [3 3]);

figure;
subplot(2,2,1), imshow(J);
subplot(2,2,2), imshow(K);
subplot(2,2,3), imshow(K1);
subplot(2,2,4), imshow(L);

% Il y a beaucoup plus de grains sur l'image à 0.08 que sur l'image à 0.02.
% L'application de imfilter sur l'image bruitée permet d'atténuer plus ou
% moins le bruit selon le niveau d'intensité du filtre. L'image résultante
% est plus floue que l'originale.
% Cependant, peu importe le bruit d'origine, on remarque que le filtre 
% `medfilt2` retourne une image quasiment similaire, non-bruitée et nette.

% La commande imfilter applique un filtre linéaire de convolution à une 
% image, où le masque de filtrage est généré par fspecial. 
% fspecial('average', n) crée un filtre moyenneur de taille n*n qui 
% remplace chaque pixel par la moyenne de ses voisins. Cela permet de 
% lisser l'image et réduire le bruit gaussien, mais avec le risque de
% flouter les contours.

% medfilt2 applique un filtre médian non-linéaire à une image. Chaque pixel
% est remplacé par la médiane des intensités des pixels de son voisinage, 
% défini par une fenêtre de taille [m,n]. Contrairement au filtre linéaire, 
% le filtre médian est efficace pour éliminer le bruit impulsionnel (comme 
% le "sel et poivre") tout en préservant les contours de l'image.