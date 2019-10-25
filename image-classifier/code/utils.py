import os
import cv2
import numpy as np
import timeit, time
from sklearn import neighbors, svm, cluster, preprocessing, metrics
from collections import defaultdict
from scipy.spatial import distance

def load_data():
    test_path = '../data/test/'
    train_path = '../data/train/'
    
    # Added code to ignore files prepended with "." such as .DS_Store
    train_classes = sorted([dirname for dirname in os.listdir(train_path) if not dirname.startswith('.')], key=lambda s: s.upper())
    test_classes = sorted([dirname for dirname in os.listdir(test_path) if not dirname.startswith('.')], key=lambda s: s.upper())
    train_labels = []
    test_labels = []
    train_images = []
    test_images = []
    for i, label in enumerate(train_classes):
        for filename in os.listdir(train_path + label + '/'):
            image = cv2.imread(train_path + label + '/' + filename, cv2.IMREAD_GRAYSCALE)
            train_images.append(image)
            train_labels.append(i)
    for i, label in enumerate(test_classes):
        for filename in os.listdir(test_path + label + '/'):
            image = cv2.imread(test_path + label + '/' + filename, cv2.IMREAD_GRAYSCALE)
            test_images.append(image)
            test_labels.append(i)
            
    return train_images, test_images, train_labels, test_labels


def KNN_classifier(train_features, train_labels, test_features, num_neighbors):
    # outputs labels for all testing images

    # train_features is an N x d matrix, where d is the dimensionality of the
    # feature representation and N is the number of training features.
    # train_labels is an N x 1 array, where each entry is an integer
    # indicating the ground truth category for each training image.
    # test_features is an M x d array, where d is the dimensionality of the
    # feature representation and M is the number of testing features.
    # num_neighbors is the number of neighbors for the KNN classifier

    # predicted_categories is an M x 1 array, where each entry is an integer
    # indicating the predicted category for each test image.

    model = neighbors.KNeighborsClassifier(n_neighbors=num_neighbors)
    model.fit(train_features, train_labels)
    predicted_categories = model.predict(test_features)

    return predicted_categories


def SVM_classifier(train_features, train_labels, test_features, is_linear, svm_lambda):
    # this function will train a linear svm for every category (i.e. one vs all)
    # and then use the learned linear classifiers to predict the category of
    # every test image. every test feature will be evaluated with all 15 svms
    # and the most confident svm will "win". confidence, or distance from the
    # margin, is w*x + b where '*' is the inner product or dot product and w and
    # b are the learned hyperplane parameters.

    # train_features is an N x d matrix, where d is the dimensionality of
    # the feature representation and N the number of training features.
    # train_labels is an N x 1 array, where each entry is an integer 
    # indicating the ground truth category for each training image.
    # test_features is an M x d matrix, where d is the dimensionality of the
    # feature representation and M is the number of testing features.
    # is_linear is a boolean. If true, you will train linear SVMs. Otherwise, you 
    # will use SVMs with a Radial Basis Function (RBF) Kernel.
    # svm_lambda is a scalar, the value of the regularizer for the SVMs

    # predicted_categories is an M x 1 array, where each entry is an integer
    # indicating the predicted category for each test feature.

    all_predictions = []
    classes = np.unique(train_labels) # All possible classes are in train_labels
    kernel = 'linear' if is_linear else 'rbf'
    for c in classes:
        b_train_labels = [1 if x == c else 0 for x in train_labels]
        clf = svm.SVC(C=svm_lambda, kernel=kernel, gamma='scale', class_weight='balanced', probability=True).fit(train_features, b_train_labels)
        predictions = clf.predict_proba(test_features)

        # Figure out what the probabilities are for class c
        c_index = np.where(clf.classes_ == 1)
        c_predictions = predictions[:,c_index]
        all_predictions.append(c_predictions)

    predicted_categories = np.argmax(all_predictions, axis=0).flatten()

    return predicted_categories

def imresize(input_image, target_size):
    # resizes the input image, represented as a 2D array, to a new image of size [target_size, target_size]. 
    # Normalizes the output image to be zero-mean, and in the [-1, 1] range.

    # Expect input_image to be a single image. 
    # Expect target size to be 8, 16, or 32
    resized_image = cv2.resize(input_image,(target_size, target_size))
    output_image = cv2.normalize(src=resized_image, dst=None, alpha=-1, beta=1, dtype=cv2.CV_32F)

    return output_image


def reportAccuracy(true_labels, predicted_labels):
    # generates and returns the accuracy of a model

    # true_labels is a N x 1 list, where each entry is an integer
    # and N is the size of the testing set.
    # predicted_labels is a N x 1 list, where each entry is an 
    # integer, and N is the size of the testing set. These labels 
    # were produced by your system.

    # accuracy is a scalar, defined in the spec (in %)

    accuracy_fraction = metrics.accuracy_score(true_labels, predicted_labels, normalize=True)
    accuracy = accuracy_fraction * 100

    return accuracy


def buildDict(train_images, dict_size, feature_type, clustering_type):
    # this function will sample descriptors from the training images,
    # cluster them, and then return the cluster centers.

    # train_images is a list of N images, represented as 2D arrays
    # dict_size is the size of the vocabulary,
    # feature_type is a string specifying the type of feature that we are interested in.
    # Valid values are "sift", "surf" and "orb"
    # clustering_type is one of "kmeans" or "hierarchical"

    # the output 'vocabulary' should be a list of length dict_size, with elements of size d, where d is the 
    # dimention of the feature. each row is a cluster centroid / visual word.

    # NOTE: Should you run out of memory or have performance issues, feel free to limit the 
    # number of descriptors you store per image.

    descriptors = []
    if feature_type == "sift":
        sift = cv2.xfeatures2d.SIFT_create(nfeatures=25)
        for image in train_images:
            _, d = sift.detectAndCompute(image, None)
            descriptors.append(d)
    elif feature_type == "surf":
        surf = cv2.xfeatures2d.SURF_create(hessianThreshold=8000)
        for image in train_images:
            _, d = surf.detectAndCompute(image, None)
            if d is None:
                continue
            descriptors.append(d)  
    else: # orb
        orb = cv2.ORB_create(nfeatures=25)
        for image in train_images:
            _, d = orb.detectAndCompute(image, None)
            if d is None:
                continue
            descriptors.append(d)

    vocabulary = []
    descriptors = np.vstack(descriptors)
    if clustering_type == "kmeans":
        clust = cluster.KMeans(n_clusters=dict_size, random_state=None).fit(descriptors)
        vocabulary = clust.cluster_centers_
    else: # hierarchical
        clust = cluster.AgglomerativeClustering(n_clusters=dict_size).fit(descriptors)
        labels = clust.labels_

        # Make a dictionary where key = label and value = array of descriptors
        dictionary = defaultdict(list)
        for k, v in zip(labels, descriptors):
            dictionary[k].append(v)
        descriptor_dict = dict(dictionary)

        for d in descriptor_dict.values():
            # Sum the columns based on @142 on Piazza
            sum_down_cols = np.sum(d, axis=0)
            # Normalize based on number of descriptors for a given label
            rows = len(d)
            avg = np.true_divide(sum_down_cols, rows)

            vocabulary.append(avg)
        vocabulary = np.array(vocabulary)

    return vocabulary


def computeBow(image, vocabulary, feature_type):
    # extracts features from the image, and returns a BOW representation using a vocabulary

    # image is 2D array
    # vocabulary is an array of size dict_size x d
    # feature type is a string (from "sift", "surf", "orb") specifying the feature
    # used to create the vocabulary

    # Get descriptors for the image
    descriptors = []
    if feature_type == "sift":
        sift = cv2.xfeatures2d.SIFT_create()
        _, d = sift.detectAndCompute(image, None)
        descriptors = d
    elif feature_type == "surf":
        surf = cv2.xfeatures2d.SURF_create()
        _, d = surf.detectAndCompute(image, None)
        descriptors = d
    else: # orb
        orb = cv2.ORB_create()
        _, d = orb.detectAndCompute(image, None)
        if d is None:
            descriptors.append(np.zeros(vocabulary.shape[1]))
        else:
            descriptors = d

    dists = distance.cdist(descriptors, vocabulary, 'euclidean')
    image_bins = np.argmin(dists, axis=1)

    Bow, _ = np.histogram(image_bins, bins=np.arange(vocabulary.shape[0] + 1), density=True)

    # BOW is the new image representation, a normalized histogram
    return Bow


def tinyImages(train_features, test_features, train_labels, test_labels):
    # Resizes training images and flattens them to train a KNN classifier using the training labels
    # Classifies the resized and flattened testing images using the trained classifier
    # Returns the accuracy of the system, and the overall runtime (including resizing and classification)
    # Does so for 8x8, 16x16, and 32x32 images, with 1, 3 and 6 neighbors

    # train_features is a list of N images, represented as 2D arrays
    # test_features is a list of M images, represented as 2D arrays
    # train_labels is a list of N integers, containing the label values for the train set
    # test_labels is a list of M integers, containing the label values for the test set

    # classResult is a 18x1 array, containing accuracies and runtimes, in the following order:
    # accuracies and runtimes for 8x8 scales, 16x16 scales, 32x32 scales
    # [8x8 scale 1 neighbor accuracy, 8x8 scale 1 neighbor runtime, 8x8 scale 3 neighbor accuracy, 
    # 8x8 scale 3 neighbor runtime, ...]
    # Accuracies are a percentage, runtimes are in seconds

    classResult = []
    image_sizes = [ 8, 16, 32 ]
    k_vals = [ 1, 3, 6 ]
    for size in image_sizes:
        # Make model and predict
        for k in k_vals:
            start = timeit.default_timer()
            train = [imresize(image, size).flatten() for image in train_features]
            test = [imresize(image, size).flatten() for image in test_features]
            predict = KNN_classifier(train, train_labels, test, k)
            stop = timeit.default_timer()
            time = stop - start

            accuracy = reportAccuracy(test_labels, predict)
            classResult.append(accuracy)
            classResult.append(time)

            print("Ran KNN on images of size {}x{} with a k of {} and got an accuracy of {}".format(size, size, k, accuracy))


    return classResult
    