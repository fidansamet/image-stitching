import os
import cv2
from matplotlib import pyplot as plt
import numpy as np
import random

plt.style.use("ggplot")
DATASET_NAME = "panorama7"
SURF = cv2.xfeatures2d.SURF_create()
# each element needs to have 2-nearest neighbors, each list of descriptors needs to have more than 2 elements each
NEAREST_NEIGHBOR_NUM = 2
RANSAC_THRESH = 2
INLIER_THRESH = 2
RANSAC_ITERATION = 1000
subsets = {}
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=100)   # or pass empty dictionary
# Matcher
bf = cv2.BFMatcher()
flann = cv2.FlannBasedMatcher(index_params, search_params)


def get_subset_names():
    root, directories, files = next(os.walk(DATASET_NAME))
    for directory in sorted(directories):
        subsets[directory] = sorted(os.listdir(DATASET_NAME + "/" + directory))


def main():
    for subset_name, subset_images_names in subsets.items():
        feature_points_plot = None
        homographies = []
        H = None
        panorama = cv2.imread(DATASET_NAME + "/" + subset_name + '/' + subset_images_names[0])
        stop = len(subset_images_names) - 1
        for i in range(0, len(subset_images_names) - 1):
            print(subset_images_names[i])

            # Read next images
            next_image = cv2.imread(DATASET_NAME + "/" + subset_name + '/' + subset_images_names[i + 1])
            # next_image = cv2.GaussianBlur(next_image, (3,3),0)
            next_image_gray = cv2.cvtColor(next_image, cv2.COLOR_BGR2GRAY)
            # next_image_gray = cv2.GaussianBlur(next_image_gray, (3,3),0)

            # Current image is the panorama
            cur_image = cv2.imread(DATASET_NAME + "/" + subset_name + '/' + subset_images_names[i])
            # cur_image = cv2.GaussianBlur(cur_image, (3,3),0)
            cur_image_gray = cv2.cvtColor(cur_image, cv2.COLOR_BGR2GRAY)
            # cur_image_gray = cv2.GaussianBlur(cur_image_gray, (3,3),0)

            # Feature extraction, feature matching, Homography finding
            homography_matrix = stitch_images(cur_image, cur_image_gray, next_image, next_image_gray, feature_points_plot)

            if homography_matrix is not None:
                homographies.append(homography_matrix)
            else:
                stop = i
                break

        # Merging by transformation
        for i in range(0, stop -1):
            print(subset_images_names[i])

            next_image = cv2.imread(DATASET_NAME + "/" + subset_name + '/' + subset_images_names[i + 1])
            cur_image = panorama

            if H is None:
                H = homographies[0]
            else:
                H = np.matmul(H, homographies[i])

            panorama = merge_images(cur_image, next_image, H)
            plt.imshow(cv2.cvtColor(panorama, cv2.COLOR_BGR2RGB))
            plt.xticks([]), plt.yticks([])
            plt.show()

        panorama = cv2.medianBlur(panorama, 3)
        plt.imshow(cv2.cvtColor(panorama, cv2.COLOR_BGR2RGB))
        plt.title('Panorama'), plt.xticks([]), plt.yticks([]), plt.show()

        # Read ground truth panorama image
        ground_truth = cv2.imread(DATASET_NAME + "/" + subset_name + '_gt.png')
        plt.imshow(cv2.cvtColor(ground_truth, cv2.COLOR_BGR2RGB))
        plt.title('Panorama Ground Truth'), plt.xticks([]), plt.yticks([]), plt.show()


def stitch_images(cur_image, cur_image_gray, next_image, next_image_gray, feature_points_plot):

    # Feature extraction
    cur_feature_pts, cur_descs, feature_points_plot = feature_extraction(cur_image, cur_image_gray, feature_points_plot)
    next_feature_pts, next_descs, feature_points_plot = feature_extraction(next_image, next_image_gray, feature_points_plot)

    # Feature matching
    matches = feature_matching(cur_image, cur_feature_pts, cur_descs, next_image, next_feature_pts, next_descs)

    # Find Homography matrix
    if matches is not None and matches is not [] and len(matches[:, 0]) >= 4:
        match_pairs_list = []
        for match in matches[:, 0]:
            (x1, y1) = next_feature_pts[match.queryIdx].pt
            (x2, y2) = cur_feature_pts[match.trainIdx].pt
            match_pairs_list.append([x1, y1, x2, y2])

        match_pairs_matrix = np.matrix(match_pairs_list)

        # Run RANSAC algorithm
        H = RANSAC(match_pairs_matrix)
        print("Final homography: ", H)
        return H

    else:
        print("Can’t find enough keypoints.")
        return None


def feature_extraction(img, img_gray, feature_points_plot):
    # Feature extraction: find the key points, compute the descriptors with SURF
    key_pts, descs = SURF.detectAndCompute(img_gray, None)
    # Plots showing feature points for each ordered pair of sub-image
    drawn_key_pts = cv2.drawKeypoints(img, key_pts, None, color=(0, 255, 0), flags=0)

    if feature_points_plot is None:
        # Assign first image's feature points
        feature_points_plot = drawn_key_pts
    else:
        # Merge first and second images' feature points
        feature_points_plot = np.concatenate((feature_points_plot, drawn_key_pts), axis=1)
        # Plot feature points of images
        plt.imshow(cv2.cvtColor(feature_points_plot, cv2.COLOR_BGR2RGB))
        plt.title('Feature Points'), plt.xticks([]), plt.yticks([])
        plt.show()

    return key_pts, descs, feature_points_plot


def feature_matching(cur_image, cur_feature_pts, cur_descs, next_image, next_feature_pts, next_descs):

    if (cur_descs is not None and len(cur_descs) > NEAREST_NEIGHBOR_NUM and
            next_descs is not None and len(next_descs) > NEAREST_NEIGHBOR_NUM):

        # Get knn detector
        matches = flann.knnMatch(next_descs, cur_descs, k=NEAREST_NEIGHBOR_NUM)

        # Apply ratio test
        good_matches = []

        for i, pair in enumerate(matches):
            try:
                m, n = pair
                if m.distance < 0.3 * n.distance:
                    good_matches.append([m])

            except ValueError:
                return None

        matches = np.asarray(good_matches)

        # cv2.drawMatchesKnn expects list of lists as matches.
        drawn_matches = cv2.drawMatchesKnn(cur_image, cur_feature_pts, next_image, next_feature_pts, good_matches, None, flags=2)    # TODO flag?
        plt.imshow(cv2.cvtColor(drawn_matches, cv2.COLOR_BGR2RGB))
        plt.title('Feature Point Matching Lines'), plt.xticks([]), plt.yticks([])
        plt.show()
        return matches

    else:
        plt.imshow(cv2.cvtColor(next_image, cv2.COLOR_BGR2RGB))
        plt.title('Feature Point Matching Lines'), plt.xticks([]), plt.yticks([])
        plt.show()
        return None


def RANSAC(match_pairs):

    max_inliers = 0
    best_H = None

    for i in range(RANSAC_ITERATION):
        # Get 4 random matches
        random_matches = np.vstack((match_pairs[random.randrange(0, len(match_pairs))],
                                match_pairs[random.randrange(0, len(match_pairs))],
                                match_pairs[random.randrange(0, len(match_pairs))],
                                match_pairs[random.randrange(0, len(match_pairs))]))

        # Get homography matrix
        H = find_homography(random_matches)
        cur_inliers = 0

        # Compute inliers where ||pi’, H pi || <ε
        for i in range(len(match_pairs)):
            d = get_geometric_distance(match_pairs[i], H)
            if d < INLIER_THRESH:
                cur_inliers += 1

        # Update maximum inlier number
        if cur_inliers > max_inliers:
            max_inliers = cur_inliers
            best_H = H

        # If exceeds threshold exit loop
        if max_inliers > (len(match_pairs) * RANSAC_THRESH):
            break

    return best_H


# TODO
def find_homography(match_pairs):
    aList = []
    for pair in match_pairs:
        p1 = np.matrix([pair.item(0), pair.item(1), 1])
        p2 = np.matrix([pair.item(2), pair.item(3), 1])
        aList.append([0, 0, 0,
              -p2.item(2) * p1.item(0), -p2.item(2) * p1.item(1), -p2.item(2) * p1.item(2),
              p2.item(1) * p1.item(0), p2.item(1) * p1.item(1), p2.item(1) * p1.item(2)])

        aList.append([-p2.item(2) * p1.item(0), -p2.item(2) * p1.item(1), -p2.item(2) * p1.item(2),
              0, 0, 0,
              p2.item(0) * p1.item(0), p2.item(0) * p1.item(1), p2.item(0) * p1.item(2)])

    matrixA = np.matrix(aList)
    u, s, v = np.linalg.svd(matrixA)
    H = np.reshape(v[8], (3, 3))
    H = (1 / H.item(8)) * H
    return H


# Calculate the geometric distance between estimated points and original points
def get_geometric_distance(match_pairs, H):
    estimated = np.dot(H, np.transpose(np.matrix([match_pairs[0].item(0), match_pairs[0].item(1), 1])))
    # Eliminate division by zero and return inefficient d
    if estimated.item(2) == 0:
        return INLIER_THRESH + 1
    err = np.transpose(np.matrix([match_pairs[0].item(2), match_pairs[0].item(3), 1])) - (1 / estimated.item(2)) * estimated
    return np.linalg.norm(err)


def merge_images(cur_image, next_image, H):
    next_image_matrix = transpose_copy(next_image, "image")
    new_row, new_col = (cur_image.shape[1] + next_image.shape[1], cur_image.shape[0])
    transformed_matrix = np.zeros((new_row, new_col, next_image_matrix.shape[2]))

    # Traverse image pixels to calculate new indices
    for i in range(next_image_matrix.shape[0]):
        for j in range(next_image_matrix.shape[1]):
            dot_product = np.dot(H, [i, j, 1])
            i_match = int(dot_product[0, 0] / dot_product[0, 2] + 0.5)
            j_match = int(dot_product[0, 1] / dot_product[0, 2] + 0.5)
            if 0 <= i_match < new_row and 0 <= j_match < new_col:
                transformed_matrix[i_match, j_match] = next_image_matrix[i, j]

    transformed_next_image = transpose_copy(transformed_matrix, "matrix")
    plt.imshow(transformed_next_image)
    plt.title(''), plt.xticks([]), plt.yticks([])
    plt.show()

    # Find non black pixels in current image and create empty mask
    non_black_mask = np.all(cur_image != [0, 0, 0], axis=-1)
    empty_mask = np.zeros((transformed_next_image.shape[0], transformed_next_image.shape[1]), dtype=bool)
    empty_mask[0:cur_image.shape[0], 0:cur_image.shape[1]] = non_black_mask

    # Assign non black pixels of current image to transformed next image
    transformed_next_image[empty_mask, :] = cur_image[non_black_mask, :]
    transformed_next_image = crop_image(transformed_next_image)
    plt.imshow(cv2.cvtColor(transformed_next_image, cv2.COLOR_BGR2RGB))
    plt.show()
    return transformed_next_image


def crop_image(image):
    # Crop top if top sum 0
    if not np.sum(image[0]):
        return crop_image(image[1:])

    # Crop bottom if bottom sum 0
    elif not np.sum(image[-1]):
        return crop_image(image[:-2])

    # Crop left if left sum 0
    elif not np.sum(image[:, 0]):
        return crop_image(image[:, 1:])

    # Crop right if right sum 0
    elif not np.sum(image[:, -1]):
        return crop_image(image[:, :-2])

    return image


def transpose_copy(copying, type):
    if type == "matrix":
        return np.transpose(copying.copy(), (1, 0, 2)).astype('uint8')
    else:
        return np.transpose(copying.copy(), (1, 0, 2))


if __name__ == '__main__':
    get_subset_names()
    main()
