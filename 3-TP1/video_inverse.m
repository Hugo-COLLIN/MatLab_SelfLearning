% Fonction d'inversion vidéo (négatif)
function I_inv = video_inverse(I)
    % Inverser l'image en soustrayant chaque pixel de 255
    I_inv = 255 - I;
end