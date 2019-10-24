# Image Classifier

This was a project I did for a Computer Vision course. 

The provided starter code is in `/starter_code`. My code is in `/image-classifier/code`. I wrote everything in `utils.py` other than `load_data`, and I wrote the code to run KNN and SVC on the images in `homework1.py`.

`/Results` contains .npy files that contains the edscriptor centroids produced by KMeans and Agglomerative Hierarchical clustering, histograms created using a bag of words approach on image descriptors, and the accuracies and runtimes of my classifiers. 

Install right version of OpenCV
```
$ pip3 uninstall opencv-python
$ pip3 install opencv-contrib-python==3.4.2.16
```