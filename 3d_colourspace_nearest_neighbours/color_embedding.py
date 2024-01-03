import cv2
import numpy as np
from glob import glob
from sklearn.preprocessing import normalize


def load_and_preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (256, 256))
    return image


def random_sample_pixels(image, sample_size=625):
    height, width, _ = image.shape
    indices = np.random.choice(height * width, size=sample_size, replace=False)
    sampled_pixels = np.array([image[idx // width, idx % width] for idx in indices])
    return sampled_pixels


def count_pixels_per_color_region(sampled_pixels, colorspace_divisions=8): #8):
    color_counts = np.zeros((colorspace_divisions, colorspace_divisions, colorspace_divisions))
    bins = np.linspace(0, 256, colorspace_divisions)
    
    for pixel in sampled_pixels:
        r, g, b = pixel.astype(int)
        
        color_counts[np.digitize(r, bins),
                     np.digitize(g, bins),
                     np.digitize(b, bins)] += 1

    return color_counts.flatten()


def image_to_color_code(image_path, sample_size=625, colorspace_divisions=4): #8):
    image = load_and_preprocess_image(image_path)
    sampled_pixels = random_sample_pixels(image, sample_size)
    return count_pixels_per_color_region(sampled_pixels, colorspace_divisions)

