import os
import cv2
from matplotlib import pyplot as plt
import numpy as np

dataset_name = "HW2_Dataset"
subsets = {}
orb = cv2.ORB_create()  # Initiate STAR detector
plt.style.use("ggplot")


def get_subset_names():
    root, directories, files = next(os.walk(dataset_name))
    for directory in sorted(directories):
        subsets[directory] = sorted(os.listdir(dataset_name + "/" + directory))


def main():
    for subset_name, subset_images_names in subsets.items():
        feature_points_plot = None
        feature_matching_plot = None
        for i in range(0, len(subset_images_names) - 1):
            print(subset_images_names[i])

            # Read current and next images
            cur_image = cv2.imread(dataset_name + "/" + subset_name + '/' + subset_images_names[i])     # TODO gray
            next_image = cv2.imread(dataset_name + "/" + subset_name + '/' + subset_images_names[i + 1])

            # Feature extraction and feature matching
            feature_points_plot, feature_matching_plot = feature_extraction_and_matching(cur_image, next_image, feature_points_plot, feature_matching_plot)

            # Find homography
            find_homography()

        #result_image = np.concatenate((feature_points_plot, cv2.cvtColor(feature_matching_plot, cv2.COLOR_BGR2RGB)), axis=0)
        plt.imshow(feature_points_plot)
        plt.title(''), plt.xticks([]), plt.yticks([])
        plt.show()


def feature_extraction_and_matching(cur_image, next_image, feature_points_plot, feature_matching_plot):

    # Feature extraction: ind the key points, compute the descriptors with ORB
    cur_key_points, cur_desc = orb.detectAndCompute(cur_image, None)
    next_key_points, next_desc = orb.detectAndCompute(next_image, None)

    # Plots showing feature points for each ordered pair of sub-image
    cur_feature_img = cv2.drawKeypoints(cur_image, cur_key_points, None, color=(0, 255, 0), flags=0)
    next_feature_img = cv2.drawKeypoints(next_image, next_key_points, None, color=(0, 255, 0), flags=0)

    if feature_points_plot is None:
        feature_points_plot = cur_feature_img
    else:
        feature_points_plot = np.concatenate((feature_points_plot, cv2.cvtColor(next_feature_img, cv2.COLOR_BGR2RGB)),
                                             axis=1)


    # Feature matching
    index_params = dict(algorithm=6,
                        table_number=6,
                        key_size=12,
                        multi_probe_level=1)
    search_params = {}

    # each element needs to have 2-nearest neighbors, each list of descriptors needs to have more than 2 elements each
    nearest_neighbor_num = 2

    # Matchers
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    if (cur_desc is not None and len(cur_desc) > nearest_neighbor_num and
            next_desc is not None and len(next_desc) > nearest_neighbor_num):

        # Get knn detector
        matches = bf.knnMatch(cur_desc, next_desc, k=nearest_neighbor_num)

        # Need to draw only good matches, so create a mask TODO use np
        matches_mask = [[0, 0] for i in range(len(matches))]

        # ratio test as per Lowe's paper
        for i, (m, n) in enumerate(matches):
            if m.distance < 0.7 * n.distance:
                matches_mask[i] = [1, 0]

        draw_params = dict(matchColor=(0, 255, 0),
                           singlePointColor=(255, 0, 0),
                           matchesMask=matches_mask,
                           flags=0)

        macthed = cv2.drawMatchesKnn(cur_image, cur_key_points, next_image, next_key_points, matches, None, **draw_params)

        if feature_matching_plot is None:
            feature_matching_plot = macthed
        else:
            feature_matching_plot = np.concatenate((feature_matching_plot, cv2.cvtColor(macthed, cv2.COLOR_BGR2RGB)),
                                                 axis=1)
    else:
        if feature_matching_plot is None:
            feature_matching_plot = next_image
        else:
            feature_matching_plot = np.concatenate((feature_matching_plot, cv2.cvtColor(next_image, cv2.COLOR_BGR2RGB)),
                                                   axis=1)

    return feature_points_plot, feature_matching_plot


def find_homography():
    return


if __name__ == '__main__':
    get_subset_names()
    main()
