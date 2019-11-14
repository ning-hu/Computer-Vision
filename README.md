# Computer Vision

These are projects I did for a Computer Vision course.

## Image Classifier

The goal of this project is to create an image classifier that given an image, can decide which class it fits into.

The provided starter code is in `starter_code`. My code is in `image-classifier/code`. I wrote everything in `utils.py` other than `load_data`, and I wrote the code to run KNN and SVC on the images in `main.py`.

`Results` contains .npy files that contains the descriptor centroids produced by KMeans and Agglomerative Hierarchical clustering, histograms created using a bag of words approach on image descriptors, and the accuracies and runtimes of my classifiers. 

Install the version of OpenCV that still has SIFT and SURF
```
$ pip3 uninstall opencv-python
$ pip3 install opencv-contrib-python==3.4.2.16
```

## Bokeh Filter

The goal of this project is to create a bokeh filter effect by synthetically enlarging the aperture of a camera by taking a video.

There was no provided starter code. Results can be found in `results.pdf` which is a modified version of `spec.pdf`. 
