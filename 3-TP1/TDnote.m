%% fonction s

function y = s(t)
    n = 1;
    y = 0;
    while n < t
        y = y + (1/2)*n;
        n = n + 1;
    end
end


s(10)
s(100)
s(1000)


%% Traitement des images



%%


%% step1
Malta = imread('Images\A_Street_in_Malta.png');
figure, histogram(Malta), title("Histogramme de l'image Street Malta");

%% step2
% imadjust : on applique imadjust à chaque composante Rouge-Vert-Bleu de
% l'image
Malta_adjust = Malta;
Malta_adjust(:,:,1) = imadjust(Malta(:,:,1));
Malta_adjust(:,:,2) = imadjust(Malta(:,:,2));
Malta_adjust(:,:,3) = imadjust(Malta(:,:,3));

% histeq
Malta_histeq = histeq(Malta);

figure;
subplot(3,2,1), imshow(Malta), title('Malta'), subplot(3,2,2), imhist(Malta);
subplot(3,2,3), imshow(Malta_adjust), title('Malta imadjust'), subplot(3,2,4), imhist(Malta_adjust);
subplot(3,2,5), imshow(Malta_histeq), title('Malta histeq'), subplot(3,2,6), imhist(Malta_histeq);


%% Otsu
level = graythresh(Malta);
BW = imbinarize(Malta, level); % im2bw est dépréciée, on utilise imbinarize à la place
titre_seuil = sprintf("Valeur du seuil : %d", level);
figure, imshowpair(Malta, BW, 'montage'), title(titre_seuil);
