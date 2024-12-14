% Fonction de soustraction de matrices avec limitation à 0

function C = soustraction(A, B)
    % Vérifier que les matrices ont la même taille
    if ~isequal(size(A), size(B))
        error('Les matrices doivent avoir la même taille');
    end
    
    % Effectuer la soustraction et limiter à 0
    C = max(A - B, 0);
end