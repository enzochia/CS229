from __future__ import division, print_function
import argparse
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import os
import random


def init_centroids(num_clusters, image):
    """
    Initialize a `num_clusters` x image_shape[-1] nparray to RGB
    values of randomly chosen pixels of`image`

    Parameters
    ----------
    num_clusters : int
        Number of centroids/clusters
    image : nparray
        (H, W, C) image represented as an nparray

    Returns
    -------
    centroids_init : nparray
        Randomly initialized centroids
    """

    # *** START YOUR CODE ***
    # raise NotImplementedError('init_centroids function not implemented')
    height, width, color = image.shape
    h_rand = [int(x) for x in np.random.uniform(0, height, num_clusters)]
    w_rand = [int(x) for x in np.random.uniform(0, width, num_clusters)]
    centroids_init = [[image[h_rand[idx], w_rand[idx], 0],
                       image[h_rand[idx], w_rand[idx], 1],
                       image[h_rand[idx], w_rand[idx], 2]]
                      for idx in range(num_clusters)]
    # *** END YOUR CODE ***

    return centroids_init


def update_centroids(centroids, image, max_iter=30, print_every=10):
    """
    Carry out k-means centroid update step `max_iter` times

    Parameters
    ----------
    centroids : nparray
        The centroids stored as an nparray
    image : nparray
        (H, W, C) image represented as an nparray
    max_iter : int
        Number of iterations to run
    print_every : int
        Frequency of status update

    Returns
    -------
    new_centroids : nparray
        Updated centroids
    """

    # *** START YOUR CODE ***
    # raise NotImplementedError('update_centroids function not implemented')
        # Usually expected to converge long before `max_iter` iterations
                # Initialize `dist` vector to keep track of distance to every centroid
                # Loop over all centroids and store distances in `dist`
                # Find closest centroid and update `new_centroids`
        # Update `new_centroids`
    height, width, color = image.shape
    num_centroids = len(centroids)
    num_iter = 0
    new_centroids = np.array([[0.0] * 3 for _ in range(num_centroids)])
    while True:
        if not (num_iter % print_every):
            print(f'Iteration #{num_iter} is in progress.')
        num_iter += 1
        cent_mat = np.array([[np.argmin([np.linalg.norm([image[idxH, idxW, 0] - centroids[idxCentroids][0],
                                                        image[idxH, idxW, 1] - centroids[idxCentroids][1],
                                                        image[idxH, idxW, 2] - centroids[idxCentroids][2]])
                                        for idxCentroids in range(len(centroids))])
                             for idxW in range(width)]
                            for idxH in range(height)])
        count_centroids = np.array([0] * num_centroids)
        for idxH in range(height):
            for idxW in range(width):
                new_centroids[cent_mat[idxH, idxW]] = np.add(new_centroids[cent_mat[idxH, idxW]],
                                                           image[idxH, idxW])
                count_centroids[cent_mat[idxH, idxW]] += 1

        for idxCentroids in range(num_centroids):
            new_centroids[idxCentroids] = np.divide(new_centroids[idxCentroids],
                                                    count_centroids[idxCentroids])

        if not np.array_equal(centroids, new_centroids) and (num_iter < max_iter):
            centroids = new_centroids
            new_centroids = np.array([[0.0] * 3 for _ in range(num_centroids)])
        else:
            break
    # *** END YOUR CODE ***

    return new_centroids


def update_image(image, centroids):
    """
    Update RGB values of pixels in `image` by finding
    the closest among the `centroids`

    Parameters
    ----------
    image : nparray
        (H, W, C) image represented as an nparray
    centroids : int
        The centroids stored as an nparray

    Returns
    -------
    image : nparray
        Updated image
    """

    # *** START YOUR CODE ***
    # raise NotImplementedError('update_image function not implemented')
            # Initialize `dist` vector to keep track of distance to every centroid
            # Loop over all centroids and store distances in `dist`
            # Find closest centroid and update pixel value in `image`
    height, width, color = image.shape
    num_centroids = centroids.shape[0]
    centroids = np.array([[int(centroids[idxCentroids, idxC])
                  for idxC in range(color)]
                 for idxCentroids in range(num_centroids)])
    cent_mat = np.array([[np.argmin([np.linalg.norm(np.add(image[idxH, idxW], -centroids[idxCentroids]))
                                     for idxCentroids in range(num_centroids)])
                          for idxW in range(width)]
                         for idxH in range(height)])
    image = np.array([[centroids[cent_mat[idxH, idxW]]
                       for idxW in range(width)]
                      for idxH in range(height)])
    # *** END YOUR CODE ***

    return image


def main(args):

    # Setup
    max_iter = args.max_iter
    print_every = args.print_every
    image_path_small = args.small_path
    image_path_large = args.large_path
    num_clusters = args.num_clusters
    figure_idx = 0

    # Load small image
    image = np.copy(mpimg.imread(image_path_small))
    print('[INFO] Loaded small image with shape: {}'.format(np.shape(image)))
    plt.figure(figure_idx)
    figure_idx += 1
    plt.imshow(image)
    plt.title('Original small image')
    plt.axis('off')
    savepath = os.path.join('.', 'orig_small.png')
    plt.savefig(savepath, transparent=True, format='png', bbox_inches='tight')

    # Initialize centroids
    print('[INFO] Centroids initialized')
    centroids_init = init_centroids(num_clusters, image)

    # Update centroids
    print(25 * '=')
    print('Updating centroids ...')
    print(25 * '=')
    centroids = update_centroids(centroids_init, image, max_iter, print_every)

    # Load large image
    image = np.copy(mpimg.imread(image_path_large))
    image.setflags(write=1)
    print('[INFO] Loaded large image with shape: {}'.format(np.shape(image)))
    plt.figure(figure_idx)
    figure_idx += 1
    plt.imshow(image)
    plt.title('Original large image')
    plt.axis('off')
    savepath = os.path.join('.', 'orig_large.png')
    plt.savefig(fname=savepath, transparent=True, format='png', bbox_inches='tight')

    # Update large image with centroids calculated on small image
    print(25 * '=')
    print('Updating large image ...')
    print(25 * '=')
    image_clustered = update_image(image, centroids)

    plt.figure(figure_idx)
    figure_idx += 1
    plt.imshow(image_clustered)
    plt.title('Updated large image')
    plt.axis('off')
    savepath = os.path.join('.', 'updated_large.png')
    plt.savefig(fname=savepath, transparent=True, format='png', bbox_inches='tight')

    print('\nCOMPLETE')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--small_path', default='./peppers-small.tiff',
                        help='Path to small image')
    parser.add_argument('--large_path', default='./peppers-large.tiff',
                        help='Path to large image')
    parser.add_argument('--max_iter', type=int, default=150,
                        help='Maximum number of iterations')
    parser.add_argument('--num_clusters', type=int, default=16,
                        help='Number of centroids/clusters')
    parser.add_argument('--print_every', type=int, default=10,
                        help='Iteration print frequency')
    args = parser.parse_args()
    main(args)
