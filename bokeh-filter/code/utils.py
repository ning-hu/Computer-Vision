import os
import cv2
import numpy as np
from skimage import data, io
from skimage.feature import match_template
from matplotlib import pyplot as plt

basepath = '../data'
framepath = basepath + '/frames'

# Loads the video of the scene and captures frames.
def load_video(video_name):

    if os.path.exists(framepath): # Remove frames folder to re-capture frames
        return load_frames()
    
    os.mkdir(framepath) 
        
    vidcap = cv2.VideoCapture(video_name)

    images = []
    count = 1
    gotframe = True
    while gotframe:
        # Get all frames in the video
        gotframe, image = vidcap.read()
        if gotframe:
            cv2.imwrite(framepath + '/image' + str(count) + '.jpg', image)
            grayimg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            images.append(grayimg)

            # plt.imshow(grayimg, cmap='gray')
            # plt.xticks([])
            # plt.yticks([])
            # plt.show()

        count += 1

    return images

# Loads the pre-captured frames.
def load_frames(gray=True):

    imagefiles = [f for f in os.listdir(framepath) if os.path.isfile(os.path.join(framepath, f)) and not f.startswith('.')]

    images = []
    for imgname in imagefiles:
        image = cv2.imread(framepath + '/' + imgname)
        if gray:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        images.append(image)

    return images

# Matches a template image to the item in the first frame of the video using 
# the cross correlation coefficient. Plots the template location in the first 
# frame of the video.
def template_match_plot(template):
    
    print("Printing image with template matching")
    baseimage = cv2.imread(basepath + '/image.jpg', cv2.IMREAD_COLOR)
    grayimg = cv2.cvtColor(baseimage, cv2.COLOR_BGR2GRAY)
    result = match_template(grayimg, template, True)

    # OpenCV reads images in BGR format instead of RGB
    image = cv2.cvtColor(baseimage, cv2.COLOR_BGR2RGB)
    img = np.array(image)

    # Get the coordinates of the template image in the scene
    ij = np.unravel_index(np.argmax(result), result.shape)
    x, y = ij[::-1]
    h, w = template.shape
    top_left = (x - int(w/2), y - int(h/2))
    bottom_right = (x + int(w/2), y + int(h/2))

    # Create a rectangle to highlight the template image in the scene
    # The search window is the entire image
    cv2.rectangle(img, top_left, bottom_right, 255, 2)
    cv2.putText(img, 'template', (top_left[0], top_left[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 2, (36,255,12), 2)
    cv2.putText(img, 'window', (h, w), cv2.FONT_HERSHEY_SIMPLEX, 2, (36,255,12), 2)
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])
    plt.show()

    return result

# Pixels shifts describe the translation of the image so that the template image
# in the frame stays in the same position across all frames.
def get_pixel_shifts(frames, template, tmpname):

    pixelshifts_x = []
    pixelshifts_y = []
    if not os.path.exists(basepath + '/pixelshifts' + tmpname + '.npy'):
        for frame in frames:
            result = match_template(frame, template, True)
            ij = np.unravel_index(np.argmax(result), result.shape)
            x, y = ij[::-1]
            pixelshifts_x.append(x)
            pixelshifts_y.append(y)

        pixelshifts = [pixelshifts_x, pixelshifts_y]
        np.save(basepath + '/pixelshifts' + tmpname + '.npy', np.asarray(pixelshifts))
        return pixelshifts

    return np.load(basepath + '/pixelshifts' + tmpname + '.npy')

def blur_image(frames, pixelshifts):

    sum_imgs = np.zeros(frames[0].shape)
    pixelshifts_x = pixelshifts[0]
    pixelshifts_y = pixelshifts[1]
    for i in range(len(frames)): 
        # Translating an image will give you negative image coordinates. You 
        # don't want that because those will not be graphed. Additionally, 
        # you can't "un-translate" those because warpAffine only takes uint8 
        # images. Therefore, you want to over-translate the original image so 
        # that once you perform the pixel shift, your entire image will have 
        # positive image coordinates. 
        rows, cols = frames[i].shape[:2]
        M = np.float32([[1, 0, pixelshifts_x[0]-pixelshifts_x[i]], [0, 1, pixelshifts_y[0]-pixelshifts_y[i]]])
        img = cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB)
        img = cv2.warpAffine(img, M, (cols, rows))
        sum_imgs = np.add(sum_imgs, img)

    sum_imgs = sum_imgs // len(frames)
    plt.imshow(sum_imgs.astype(np.uint8))
    plt.xticks([])
    plt.yticks([])
    plt.show()