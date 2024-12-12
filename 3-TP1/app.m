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

%% > Q3
J = imnoise(Ing, 'salt & pepper', 0.08);
K = imfilter(J, fspecial('average', 5));
K1 = imfilter(J, fspecial("average", 9));
L = medfilt2(J, [3 3]);

figure;
subplot(2,2,1), imshow(J);
subplot(2,2,2), imshow(K);
subplot(2,2,3), imshow(K1);
subplot(2,2,4), imshow(L);

% Q3: Il y a beaucoup plus de grains sur l'image à 0.08 que sur l'image à 0.02.
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


%% > Q4

Jg = imnoise(I, 'gaussian', 0, 0.01);

h1 = fspecial('gaussian', [5 5], 1); % Filtre gaussien avec sigma = 1
h3 = fspecial('gaussian', [7 7], 3); % Filtre gaussien avec sigma = 3

Jg_sigma1 = imfilter(Jg, h1, 'replicate');
Jg_sigma3 = imfilter(Jg, h3, 'replicate');

figure;
subplot(1,3,1), imshow(Jg), title('Image avec bruit gaussien');
subplot(1,3,2), imshow(Jg_sigma1), title('Filtrée avec sigma = 1');
subplot(1,3,3), imshow(Jg_sigma3), title('Filtrée avec sigma = 3');



% Q4: Avec un sigma faible, les contours sont mieux préservés, tandis 
% qu’avec un sigma élevé la réduction du bruit est plus importante au prix 
% d'un floutage accru.


%% > Q5

% Créer le filtre laplacien avec alpha = 0.2
h_laplacian = fspecial('laplacian', 0.2);

% Appliquer le filtre laplacien à l'image Ing
J_laplacian = imfilter(Ing, h_laplacian, 'replicate');

% Afficher l'image résultante
figure;
imshow(J_laplacian);
title('Image filtrée avec un filtre Laplacien (alpha = 0.2)');

%Q5: On observe uniquement les contours en blanc de l'objet représenté
%(l'hélicoptère) sur fond noir.


%% > Q6

% Définir le masque personnalisé
h_custom = [-0.25 -1 -0.25; 
            -1    6.5 -1; 
            -0.25 -1 -0.25];

% Appliquer le filtre personnalisé à l'image Ing
J_custom = imfilter(Ing, h_custom, 'replicate');

% Afficher l'image résultante
figure;
imshow(J_custom);
title('Image filtrée avec le masque personnalisé');

% Q6: L'image apparait plus nette, tranchée, elle est plus contrastée.

% Pas de Q7 ?

%% II.4. Segmentation image
%% > Méthode du K-means
[L5,Centers5] = imsegkmeans(Ing,5);
[L3,Centers3] = imsegkmeans(Ing,3);

B = labeloverlay(Ing, L5);
B1 = labeloverlay(Ing, L3);

figure;
subplot(1,2,1), imshow(B), title('Segmentation 5');
subplot(1,2,2), imshow(B1); title('Segmentation 3');

% Q8: On segmente les images, c'est-à-dire que l'on essaie de les 4
% représenter avec un nombre limité de couleurs. La segmentation à 5
% couleurs  (image B) donne une image qui ressemble plus à l'originale que
% la segmentation à 3 couleurs (image B1).

%% > Méthode Otsu
level = graythresh(Ing);
BW = imbinarize(Ing, level);
figure, imshowpair(Ing, BW, 'montage'), title(level);


% Q9: level = 0.24902
% Tandis que K-means divise l'image en K clusters (ici 3 ou 5) basés sur 
% les intensités des pixels, Otsu trouve un seuil unique qui minimise la 
% variance intra-classe.

%% II. 5. Détection de contours
B1 = edge(Ing, 'sobel');
B2 = edge(Ing, 'canny');
figure, imshowpair(B1, B2, "montage");

%% > Q10
B2_modified = edge(Ing, 'canny', 0.1, 3); % Détection de contours avec Canny et paramètres modifiés

figure;
imshow(B2_modified);
title('Détection de contours avec Canny (sigma=3, seuil=0.1)');

%%

% Détection de contours avec différents seuils Sobel
B1 = edge(Ing, 'sobel');                  % Seuil par défaut
B2 = edge(Ing, 'sobel', 0.1);             % Seuil à 0.1
B3 = edge(Ing, 'sobel', 0.2);             % Seuil à 0.2

% Affichage des résultats
figure;
subplot(1,3,1), imshow(B1), title('Sobel - Seuil par défaut');
subplot(1,3,2), imshow(B2), title('Sobel - Seuil 0.1');
subplot(1,3,3), imshow(B3), title('Sobel - Seuil 0.2');

% Q10: Le principe de Sobel repose sur le calcul des gradients horizontaux 
% et verticaux via deux matrices de convolution. En variant le seuil, on 
% contrôle la sensibilité de la détection de contours : un seuil bas (0.1) 
% conserve plus de détails et de petits contours, tandis qu'un seuil plus 
% élevé (0.2) ne retient que les contours les plus marqués, réduisant ainsi 
% le bruit et les détails secondaires.


%% > Q11
% Calcul des passages par zéros avec LoG pour sigma = 1 et sigma = 3
log1 = edge(Ing, 'log', 0, 1);
log3 = edge(Ing, 'log', 0, 3);

% Rappel : Calcul des contours de Canny
B2_canny = edge(Ing, 'canny');

% Affichage des résultats
figure;
subplot(1,3,1), imshow(log1), title('LoG - sigma = 1');
subplot(1,3,2), imshow(log3), title('LoG - sigma = 3');
subplot(1,3,3), imshow(B2_canny), title('Contours Canny');

% Q11: La méthode LoG (Laplacian of Gaussian) permet de détecter les 
% passages par zéros avec deux paramètres de sigma différents, révélant 
% des comportements distincts. Avec un sigma de 1, la détection capture 
% des contours très fins et détaillés, mettant en évidence les 
% micro-variations de l'image, mais restant plus sensible au bruit. 
% 
% En augmentant sigma à 3, les contours deviennent plus larges et lissés. 
% Cela réduit significativement les détails secondaires et le bruit, ce qui 
% permet de mettre en évidence les contours principaux de manière plus 
% nette. 
% 
% En comparaison, la méthode de Canny apparaît globalement plus 
% performante : elle offre des contours plus propres et mieux définis grâce à 
% son algorithme plus sophistiqué de suppression des contours parasites. 
% 
% Le choix entre ces méthodes dépendra donc de la nature de l'image et de 
% l'objectif recherché : précision des détails avec LoG (sigma 1) ou 
% robustesse et netteté globale avec Canny ou LoG (sigma 3).


%% II.6. Détection des coins-corrélations
%% > Q12
% Détection de coins avec différentes méthodes
% Harris
points_harris = detectHarrisFeatures(Ing);
strongest_harris = points_harris.selectStrongest(20);

% FAST
points_fast = detectFASTFeatures(Ing);
strongest_fast = points_fast.selectStrongest(20);

% Corner (méthode traditionnelle)
corners_traditional = corner(Ing);
strongest_corners = corner(Ing, 'QualityLevel', 0.95);  % Réduction des faux points

% Affichage des coins
figure;
subplot(1,3,1), imshow(Ing), title('Harris Features');
hold on;
plot(strongest_harris);
hold off;

subplot(1,3,2), imshow(Ing), title('FAST Features');
hold on;
plot(strongest_fast);
hold off;

subplot(1,3,3), imshow(Ing), title('Corner Features (réduits)');
hold on;
plot(strongest_corners, 'r*');
hold off;

% Q12: La méthode de Harris semble efficace pour identifier les coins 
% les plus significatifs de l'image. Les résultats sont stables et peu 
% sensibles au bruit. 
% La méthode FAST est plus rapide d'exécution mais est plus sensible aux 
% détails fins de l'image. 
% La méthode Corner traditionnelle donne des résultats totalement erronés.
% 
% Pour améliorer les résultats de la méthode Corner et réduire les faux 
% positifs, on peut augmenter le seuil de qualité à 0,95. Cela permet de se
% concentrer sur les coins les plus importants et de réduire 
% significativement le nombre de faux positifs. 
% On peut aussi limiter le nombre de coins détectés, comme c'est le cas
% pour les méthodes Harris et FAST.


%% II.7. Correlation
%% > Q13
%% >> 1. Crop Hobby lobby et effectuer la corrélation
% Isoler "Hobby Lobby" avec imcrop
pattern = imcrop(Ing, [560 155 70 20]);
figure, subplot(1,2,1), imshow(pattern), subplot(1,2,2), imshow(Ing);


% Réaliser la corrélation normalisée
c = normxcorr2(pattern, Ing);

% Afficher la nappe 3D de corrélation
figure;
surf(c);
title('Surface de Corrélation Croisée');
zlabel('Valeur de Corrélation');


%% >> Chercher le maximum de corrélation
% Trouver le maximum de corrélation
[max_val, max_idx] = max(c(:));
[vert_coord, horiz_coord] = ind2sub(size(c), max_idx);

% Ajuster les coordonnées pour correspondre à l'image originale
offset_x = size(pattern, 2) / 2;
offset_y = size(pattern, 1) / 2;
match_x = horiz_coord - offset_x;
match_y = vert_coord - offset_y;

fprintf('(%d,%d)\n', xpeak, ypeak)

% Marquer la position sur l'image originale
figure;
imshow(Ing);
hold on;
plot(match_x, match_y, 'r*', 'MarkerSize', 10);
title('Position du Motif Détecté');

% Q13: Le maximum de corrélation calculé est (x=630,y=175)

%% II.8. Morpho-math
%% > Q14
morpho_img = imread('Images\Morpho.tif');
% figure, imshow(morpho_img);

% Définir l'élément structurant carré
SE = ones(3,3);

% Réaliser la dilatation
dilated_img = imdilate(morpho_img, SE);

figure;
subplot(1,2,1), imshow(morpho_img), title('Image originale');
subplot(1,2,2), imshow(dilated_img), title('Image dilatée');

% Q14: On observe que la dilatation permet d'étendre légèrement les zones
% blanches de l'image, mais ne ferme pas entièrement les trous.
% Pour fermer complètement les trous, on pourrait :
% - Augmenter la taille de l'élément structurant (par exemple, utiliser un carré 5x5 ou plus grand).
% - Appliquer plusieurs fois l'opération de dilatation.

%% > Q15
% Réaliser l'érosion
eroded_img = imerode(morpho_img, SE);

% Afficher les images
figure;
subplot(1,2,1), imshow(morpho_img), title('Image originale');
subplot(1,2,2), imshow(eroded_img), title('Image érodée');

% Q15: L'érosion a l'effet inverse de la dilatation. On constate que les 
% zones blanches de l'image sont réduites, tandis que les trous noirs sont 
% agrandis. Cette opération accentue les détails fins et peut séparer des 
% objets connectés, mais elle ne permet pas de fermer les trous.
% 
% (L'érosion a un effet inverse à la dilatation. Elle réduit les zones 
% blanches de l'image en retirant les pixels en périphérie des objets. On 
% observe un rétrécissement des objets blancs, avec une perte de détails et
% une diminution de leur taille. Les contours deviennent plus fins et 
% certains petits objets disparaissent complètement. 
% Cela permet d'amincir et de simplifier la structure des objets dans
% l'image.)


%% > Q16
% Dilatation suivie d'érosion (fermeture)
closed_img = imerode(imdilate(morpho_img, SE), SE);

% Érosion suivie de dilatation (ouverture)
opened_img = imdilate(imerode(morpho_img, SE), SE);

% Afficher les résultats
figure;
subplot(1,3,1), imshow(morpho_img), title('Image originale');
subplot(1,3,2), imshow(closed_img), title('Fermeture');
subplot(1,3,3), imshow(opened_img), title('Ouverture');


% Q16: On remarque que :
% - La dilatation suivie de l'érosion (appelé fermeture) a fermé certains des petits trous de l'image, tout en préservant la forme générale des objets.
% - L'érosion suivie de la dilatation (appelé ouverture) a lissé les contours des objets et supprimé certains détails fins.
% Finalement, nous avons réalisé deux opérations morphologiques fondamentales :
% - La fermeture (closing) : utile pour combler les petits trous et les fissures étroites dans les objets.
% - L'ouverture (opening) : utile pour éliminer les petites protubérances et lisser les contours des objets.



%% II.9. Reconnaissance de formes
%% > Q17
% Créer une image carrée de 100x100 pixels
I = zeros(100,100);
I(25:75, 25:75) = 1;

% Calculer la transformée de Radon
theta = 0:180;
[R,xp] = radon(I,theta);

% Affichée la transformée de radon
figure
iptsetpref('ImshowAxesVisible','on'); % Afficher l'échelle des axes sur l'image
imshow(R,[],'Xdata',theta,'Ydata',xp,'InitialMagnification','fit')
xlabel('\theta (degrees)')
ylabel('x''')
colormap(gca,hot), colorbar
iptsetpref('ImshowAxesVisible','off'); % Masquer l'échelle des axes

% Q17: La fonction radon calcule la transformée de Radon d'une image en 
% projetant celle-ci le long de rayons orientés selon différents angles. 
% Pour chaque angle, elle somme les intensités des pixels le long de lignes
% perpendiculaires à la direction de projection.




%% > Q18
% Calcul de la signature : somme des carrés de chaque colonne
function signature = calculerSignature1D(R)
    % R est la matrice de Radon
    signature = sum(R.^2, 1);
end

% Création de l'image carrée
I = zeros(100);
I(25:75, 25:75) = 1;

% Calcul de la transformée de Radon
theta = 0:180;
[R,xp] = radon(I,theta);

% Calcul de la signature 1D
signature = calculerSignature1D(R);

% Afficher la signature
figure;
plot(theta, signature);
title('Signature 1D');
xlabel('\theta (degrés)');
ylabel('Intensité');

% Affichage de l'image originale, de la transformée de Radon et de la signature
% figure;
% subplot(3,1,1);
% imshow(I);
% title('Image originale');
% 
% subplot(3,1,2);
% iptsetpref('ImshowAxesVisible','on'); % Afficher l'échelle des axes sur l'image
% imshow(R,[],'Xdata',theta,'Ydata',xp,'InitialMagnification','fit')
% title('Transformée de Radon')
% xlabel('\theta (degrees)')
% ylabel('x''')
% colormap(gca,hot), colorbar
% iptsetpref('ImshowAxesVisible','off'); % Masquer l'échelle des axes
% 
% subplot(3,1,3);
% plot(theta, signature);
% title('Signature 1D');
% xlabel('\theta (degrés)');
% ylabel('Intensité');


%% > Q19
I_rotated = imrotate(I, 56, 'crop');
I_resized = imresize(I, 1.25, 'nearest');

% Calcul des transformées de Radon et des signatures
[R_rotated, ~] = radon(I_rotated, theta);
[R_resized, ~] = radon(I_resized, theta);

signature_rotated = calculerSignature1D(R_rotated);
signature_resized = calculerSignature1D(R_resized);

% Affichage des signatures
figure;
plot(theta, signature, 'b', theta, signature_rotated, 'r', theta, signature_resized, 'g');
legend('Original', 'Tourné 56°', 'Zoom 1.25');
xlabel('\theta (degrés)');
ylabel('Intensité');
title('Comparaison des signatures 1D');


% Q19: 
% - La signature du carré original présente des pics importants à 0°, 90° et 180°
% correspondant aux côtés du carré, des pics très faibles à 45° et 135°, 
% ainsi que des creux à 30°, 60°, 120° et 150°
% correspondant aux diagonales.
% - Pour le carré tourné à 56°, les pics et creux sont décalés de 
% 56° par rapport à la signature du carré d'origine, tout en conservant 
% la forme générale de la signature.
% - Pour le carré zoomé à 1.25, la forme générale de la signature est
% identique à celle du carré original, mais l'intensité moyenne est
% beaucoup plus importante (facteur 2.5 environ).

% On en retire donc que :
% - La rotation de l'image se traduit par un décalage de la signature dans l'espace angulaire.
% - Le changement d'échelle (zoom) affecte principalement l'amplitude de la signature, tout en préservant sa forme générale.
% On en conclut donc que la signature 1D basée sur la transformée de Radon est invariante par translation, mais sensible aux rotations et aux changements d'échelle.



%% > Q20 : Calculer la R-signature d'un disque

% 1-Créez une image contenant un disque blanc :
I = zeros(100, 100,3);
center = [50, 50];
radius = 25;
I = insertShape(I, 'FilledCircle', [center, radius], 'Color', 'white', 'Opacity', 1);
I = rgb2gray(I);
I = im2double(I);

% 2-Calculez la transformée de Radon
theta = 0:180;
[R, xp] = radon(I, theta);

% 3-Calculez la R-signature
signature = calculerSignature1D(R);

% Affichage
figure;
subplot(2,1,1);
imshow(I);
title('Image du disque');

subplot(2,1,2);
plot(theta, signature);
title('R-signature du disque');
xlabel('\theta (degrés)');
ylabel('Intensité');


% Q20: Dans un cercle parfait, la signature devrait être constante pour tous
% les angles. Cependant, insertShape produit un cercle de basse résolution, 
% ce qui entraine des variations dans la signature, en particulier des pics 
% hauts à 45°, 135° et un pic bas à 90° (8.4563*10^4).


%%

% (La solution la plus efficace pour fermer les trous de l'image serait 
% d'utiliser la fermeture morphologique (fonction imclose()), une technique 
% qui combine une dilatation suivie d'une érosion. Cette méthode permet de 
% combler les petits trous et les connexions entre objets proches.)

%% >> Fermeture morphologique
SE_close = strel('square', 3);  % Élément structurant légèrement plus grand
closed_img = imclose(morpho_img, SE_close);

figure;
subplot(1,2,1), imshow(morpho_img), title('Image originale');
subplot(1,2,2), imshow(closed_img), title('Image après fermeture');




%%
imshow(Ing)