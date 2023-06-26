# Import needed packages

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, MiniBatchKMeans
from PIL import Image
import cv2
import os
import random
from tqdm import tqdm


def Silhouette(data, labels, centers):
    num_data = data.shape[0]
    k = centers.shape[0]

    a = np.zeros((num_data, 1))
    b = np.zeros((num_data, 1))
    s = np.zeros((num_data, 1))
    for i in range(num_data):  # Iterate through each image
        cluster = labels[i]
        d = list()

        distance = np.zeros((num_data, 1))

        # Perform the calculation in batches for improved speed
        indx = 0
        for j in range(0, num_data, 100000):
            dist = np.square(data[indx:j,:] - data[i,:].astype('int32'))
            distance[indx:j] = np.sum(dist, axis=1, keepdims=True)
            indx = j
        dist = np.square(data[indx:, :] - data[i,:].astype('int32'))
        distance[indx:] = np.sum(dist, axis=1, keepdims=True)

        distance = np.sqrt(distance)
        for j in range(k):  # Iterate through each cluster
            cluster_j_indx = (labels == j)
            cluster_j_indx[i] = False
            # num_cluster_j_data = np.sum(cluster_j_indx)
            # cluster_j_data = data[cluster_j_indx, :]

            distances = distance[cluster_j_indx]
            if (j == cluster):  # Same cluster as image
                a[i] = np.mean(distances)
            else:  # a different cluster
                d.append(np.mean(distances))

        b[i] = np.min(np.array(d))

    denom = np.maximum(a, b)
    s = (b - a) / denom
    return np.mean(s)


if __name__ == "__main__":
    # The base directory to images
    DATA_DIR = '../../../In-situ Meas Data/In-situ Meas Data/Melt Pool Camera Preprocessed PNG/'

    # The size to transform the images to
    IMG_SIZE = 50

    # The range of clusters to test
    MIN_K = 2
    MAX_K = 3

    # Load the preprocessed data
    meltpools = np.load("K_Means_Meltpools.npy")
    meltpools_name = np.load("K_Means_Meltpools_Name.npy")

    clusters = np.load("clusters.npy")
    centroids = np.load("centroids.npy")

    Silhouette(meltpools, clusters, centroids)