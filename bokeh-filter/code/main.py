import os
import numpy as np
from skimage import data, io
from skimage.feature import match_template
from matplotlib import pyplot as plt

from utils import *

if __name__ == "__main__":

    basepath = '../data'

    print("Loading video and extracting frames in grayscale")
    frames = load_video(basepath + '/video.mp4')

    templates = ['template', 'template2']
    for tmpname in templates:
        template = cv2.imread(basepath + '/' + tmpname + '.jpg', cv2.IMREAD_GRAYSCALE)
        matrix = template_match_plot(template)

        if not os.path.exists(basepath + '/xcc.jpg'):
            print("Printing the cross correlation coefficient matrix")
            plt.imshow(matrix, cmap='gray')
            plt.xlabel("Pixel location in X direction")
            plt.ylabel("Pixel location in Y direction")
            plt.colorbar(fraction=0.046, pad=0.04)
            plt.show()

        print("Retrieving pixel shifts")
        pixelshifts = get_pixel_shifts(frames, template, tmpname)
        
        if not os.path.exists(basepath + '/pixelshifts.jpg'):
            plt.scatter(pixelshifts[0], pixelshifts[1])
            plt.xlabel("Y Pixel Shift")
            plt.ylabel("X Pixel Shift")
            plt.show()

        print("Blurring image")
        color_frames = load_frames(False)
        blur_image(color_frames, pixelshifts)
