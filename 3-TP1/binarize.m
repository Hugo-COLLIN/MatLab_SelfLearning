% Binariser une matrice selon un seuil
function binarized = binarize(matrice, s)
	binarized = (matrice >= s);
end