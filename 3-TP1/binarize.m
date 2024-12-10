% Binarize the input image using the given threshold

function binarized_image = binarize(image, threshold)

binarized_image = image;
binarized_image(image<threshold) = 0;
binarized_image(image>threshold) = 255;

end