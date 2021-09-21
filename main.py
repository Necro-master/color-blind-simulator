import numpy as np
from PIL import Image
import sys

T = np.array([
    [0.3904725, 0.54990437, 0.00890159],
    [0.07092586, 0.96310739, 0.00135809],
    [0.02314268, 0.12801221, 0.93605194]
])

rgb_to_lms = T

lms_to_rgb = np.linalg.inv(T)

lms_protanopia = np.array([[0, 0.90822864, 0.008192], [0, 1, 0], [0, 0, 1]])

lms_deutraponia = np.array([[1, 0, 0], [1.10104433,  0, -0.00901975], [0, 0, 1]])

lms_tritanopia = np.array([[1, 0, 0], [0, 1, 0], [-0.15773032,  1.19465634, 0]])

def simulate_protanopia(image_lms, name, format):
    img = np.einsum("ij, ...j", lms_protanopia, image_lms)
    save_image(img, name + '_protanopia' + format)

def simulate_deutranopia(image_lms, name, format):
    img = np.einsum("ij, ...j", lms_deutraponia, image_lms)
    save_image(img, name + '_deutranopia' + format)

def simulate_tritanopia(image_lms, name, format):
    img = np.einsum("ij, ...j", lms_tritanopia, image_lms)
    save_image(img, name + '_tritanopia' + format)

def save_image(image, name):
    new_image = np.einsum("ij, ...j", lms_to_rgb, image)
    Image.fromarray(new_image.astype('uint8')).save(name)

def main():
    full_name = sys.argv[1]
    name = full_name.split('.')[0]
    format = '.' + full_name.split('.')[1]
    img_rgb = np.array(Image.open(name + format).convert('RGB'))
    img_lms = np.einsum("ij, ...j", rgb_to_lms, img_rgb)
    simulate_tritanopia(img_lms, name, format)
    simulate_deutranopia(img_lms, name, format)
    simulate_protanopia(img_lms, name, format)

if __name__ == '__main__':
    main()