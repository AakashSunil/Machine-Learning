import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
import scipy
import numpy.matlib


def closest_centroids(X, centroids):
    m = X.shape[0]
    k = centroids.shape[0]
    idx = np.zeros(m)
    
    for i in range(m):
        min_dist = 1000000
        for j in range(k):
            dist = np.sum((X[i,:] - centroids[j,:]) ** 2)
            if dist < min_dist:
                min_dist = dist
                idx[i] = j
    
    return idx

def centroid_computation(X, idx, k):
    n = X.shape[1]
    centroids = np.zeros((k, n))
    
    for i in range(k):
        indices = np.where(idx == i)
        centroids[i,:] = (np.sum(X[indices,:], axis=1) / len(indices[0])).ravel()
    
    return centroids

def k_means_algo(X, initial_centroids):

    # Iterations to update the moving centroids - general case - usually the centroid does not move after about 10 iterations
    max_iterations = 10
    m = X.shape[0]
    k = initial_centroids.shape[0]
    idx = np.zeros(m)
    centroids = initial_centroids
    
    for i in range(max_iterations):
        # Finding the closest centroid in the cluster to shift the previous centroids
        idx = closest_centroids(X, centroids)

        # Update the centroid location 
        centroids = centroid_computation(X, idx, k)
    
    return idx, centroids

def initial_centorid_computation(X, k):
    # Shape of the Image Array 
    m, n = X.shape

    # Initializing centroids to zeros
    centroids = np.zeros((k, n))

    # Random points taken as centroid location
    idx = np.random.randint(0, m, k)
    for i in range(k):
        centroids[i,:] = X[idx[i],:]
    
    return centroids

def kmeans(X,k):

    # Initialize Random Centriod Points in the image clusters
    initial_centroids = initial_centorid_computation(X,k)

    #Run the K_Means algothim - Returns Centroids Array with indexes of the modified Centroids
    idx, centroids = k_means_algo(X, initial_centroids)

    # Moving the centroid to the closest 
    idx = closest_centroids(X, centroids)

    # Compressed Image integer array
    X_recovered = centroids[idx.astype(int),:]

    # Converting 1d Array to 2d image compressed image
    X_recovered = np.reshape(X_recovered, (A.shape[0], A.shape[1], A.shape[2]))

    return X_recovered

print(len(sys.argv))
if(len(sys.argv)<4):
    print('Missing Arguments')
    print('Program Usage: python k_means.py <Input Image> <K Value> <Output Image>')
else:
    k_means = [2,5,10,15,20]
    original_image = sys.argv[1]
    k = int(sys.argv[2])
    output_compressed_image = sys.argv[3]
    input_img = mpimg.imread(original_image)

    fig = plt.figure()
    a = fig.add_subplot(1,2,1)
    input_imgplot = plt.imshow(input_img)
    
    #Convert image points to array
    A = np.array(input_img)
    #Normalize the data points to be in th [0,1] range
    A = A / 255
    # Flatten the 2d data points
    X = np.reshape(A,(A.shape[0] * A.shape[1], A.shape[2]))

    # Call the k-means Algorithm
    output_kmeans = kmeans(X,k)

    #Saving the compressed Image in the output file
    mpimg.imsave(output_compressed_image,output_kmeans)

    # Display the Output Compressed Image
    output_img = mpimg.imread(output_compressed_image)
    a = fig.add_subplot(1,2,2)
    output_imgplot = plt.imshow(output_img)
    # Reading input and output images to calculate compression ratios 
    original_file_size = os.stat(original_image).st_size
    compressed_file_size = os.stat(output_compressed_image).st_size

    # Compression Ratio
    compression = original_file_size/compressed_file_size

    print('Compression Ratio for K=',k,': ',compression)
    plt.show()