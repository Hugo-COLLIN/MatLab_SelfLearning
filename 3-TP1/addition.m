% Fonction d'addition de matrices avec limitation à 255

function C = addition(A, B)
    % Vérifier que les matrices ont la même taille
    if ~isequal(size(A), size(B)) 
        error('Les matrices doivent avoir la même taille');
    end
    
    % Effectuer l'addition et limiter à 255
    C = min(A + B, 255);
end
