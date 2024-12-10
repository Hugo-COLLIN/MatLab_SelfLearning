function image = create_image(val_fond, val_pattern, height, width, xh, yh, xb, yb)
    % Créer une image de la taille demandée avec le fond de la couleur val_fond
    image = ones(height, width) * val_fond;
    
    % Ajouter le rectangle de la couleur val_pattern aux coordonnées (xh, yh) et (xb, yb)
    image(yh:yb, xh:xb) = val_pattern;
    
    % Renvoyer l'image résultante
end