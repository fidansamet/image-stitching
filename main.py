import os
import cv2
from matplotlib import pyplot as plt
import numpy as np
import random

plt.style.use("ggplot")
DATASET_NAME = "HW2_Dataset"
ORB = cv2.ORB_create()  # Initiate ORB detector
# each element needs to have 2-nearest neighbors, each list of descriptors needs to have more than 2 elements each
NEAREST_NEIGHBOR_NUM = 2
RANSAC_THRESH = 2
INLIER_THRESH = 2
subsets = {}


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
            homographies = stitch_images(cur_image, cur_image_gray, next_image, next_image_gray, feature_points_plot, homographies)

        # Merging by transformation
        for i in range(0, len(subset_images_names) - 1):
            print(subset_images_names[i])

            next_image = cv2.imread(DATASET_NAME + "/" + subset_name + '/' + subset_images_names[i + 1])
            cur_image = panorama

            if H is None:
                H = homographies[0]
            else:
                H = np.matmul(H, homographies[i])

            panorama = merge_images(cur_image, next_image, H)
            plt.imshow(panorama)
            plt.show()

        panorama = cv2.medianBlur(panorama, 3)
        plt.imshow(panorama)
        plt.title('Panorama'), plt.xticks([]), plt.yticks([]), plt.show()

        # Read ground truth panorama image
        ground_truth = cv2.imread(DATASET_NAME + "/" + subset_name + '_gt.png')
        plt.imshow(ground_truth)
        plt.title('Panorama Ground Truth'), plt.xticks([]), plt.yticks([]), plt.show()


def stitch_images(cur_image, cur_image_gray, next_image, next_image_gray, feature_points_plot, homographies):

    # Feature extraction
    cur_feature_pts, cur_descs, feature_points_plot = feature_extraction(cur_image, cur_image_gray, feature_points_plot)
    next_feature_pts, next_descs, feature_points_plot = feature_extraction(next_image, next_image_gray, feature_points_plot)

    # Feature matching
    matches = feature_matching(cur_image, cur_feature_pts, cur_descs, next_image, next_feature_pts, next_descs)

    # Find Homography matrix
    if len(matches[:, 0]) >= 4:
        match_pairs_list = []
        if matches is not None or matches is not []:
            for match in matches[:, 0]:
                (x1, y1) = next_feature_pts[match.queryIdx].pt
                (x2, y2) = cur_feature_pts[match.trainIdx].pt
                match_pairs_list.append([x1, y1, x2, y2])

        match_pairs_matrix = np.matrix(match_pairs_list)

        # Run RANSAC algorithm
        H = RANSAC(match_pairs_matrix, RANSAC_THRESH)

        homographies.append(H)
        print("Final homography: ", H)

    else:
        print("Can’t find enough keypoints.")

    return homographies


def feature_extraction(img, img_gray, feature_points_plot):
    # Feature extraction: find the key points, compute the descriptors with ORB
    key_pts, descs = ORB.detectAndCompute(img_gray, None)
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

    # Matcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    index_params = dict(algorithm=6,
                        table_number=6,
                        key_size=12,
                        multi_probe_level=2)
    search_params = {}
    flann = cv2.FlannBasedMatcher(index_params, search_params)


    if (cur_descs is not None and len(cur_descs) > NEAREST_NEIGHBOR_NUM and
            next_descs is not None and len(next_descs) > NEAREST_NEIGHBOR_NUM):

        # Get knn detector
        matches = flann.knnMatch(next_descs, cur_descs, k=NEAREST_NEIGHBOR_NUM)

        # Apply ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < 0.8 * n.distance:  # TODO
                good_matches.append([m])
        matches = np.asarray(good_matches)

        # cv2.drawMatchesKnn expects list of lists as matches.
        drawn_matches = cv2.drawMatchesKnn(cur_image, cur_feature_pts, next_image, next_feature_pts, good_matches, None, flags=2)    # TODO flag?
        plt.imshow(cv2.cvtColor(drawn_matches, cv2.COLOR_BGR2RGB))
        plt.title('Feature Point Matching Lines'), plt.xticks([]), plt.yticks([])
        plt.show()
        return matches

    else:
        # TODO
        plt.imshow(cv2.cvtColor(next_image, cv2.COLOR_BGR2RGB))
        plt.title('Feature Point Matching Lines'), plt.xticks([]), plt.yticks([])
        plt.show()
        return None


def RANSAC(match_pairs, thresh):

    max_inliers = 0
    best_H = None

    for i in range(100):

        # Find 4 feature (random) points to calculate a homography
        corr1 = match_pairs[random.randrange(0, len(match_pairs))]
        corr2 = match_pairs[random.randrange(0, len(match_pairs))]
        randomFour = np.vstack((corr1, corr2))
        corr3 = match_pairs[random.randrange(0, len(match_pairs))]
        randomFour = np.vstack((randomFour, corr3))
        corr4 = match_pairs[random.randrange(0, len(match_pairs))]
        randomFour = np.vstack((randomFour, corr4))

        # Compute homography
        H = find_homography(randomFour)
        inliers = 0

        # Compute inliers where ||pi’, H pi || <ε
        for i in range(len(match_pairs)):
            d = least_squares(match_pairs[i], H)
            if d < INLIER_THRESH:
                inliers += 1

        # Update inlier number
        if inliers > max_inliers:
            max_inliers = inliers
            best_H = H

        print("Corr size: ", len(match_pairs), " NumInliers: ", inliers, "Max inliers: ", max_inliers)

        if max_inliers > (len(match_pairs) * thresh):
            break

    return best_H


def find_homography(matches):
    # loop through correspondences and create assemble matrix
    aList = []
    for corr in matches:
        p1 = np.matrix([corr.item(0), corr.item(1), 1])
        p2 = np.matrix([corr.item(2), corr.item(3), 1])

        a2 = [0, 0, 0, -p2.item(2) * p1.item(0), -p2.item(2) * p1.item(1), -p2.item(2) * p1.item(2),
              p2.item(1) * p1.item(0), p2.item(1) * p1.item(1), p2.item(1) * p1.item(2)]

        a1 = [-p2.item(2) * p1.item(0), -p2.item(2) * p1.item(1), -p2.item(2) * p1.item(2), 0, 0, 0,
              p2.item(0) * p1.item(0), p2.item(0) * p1.item(1), p2.item(0) * p1.item(2)]

        aList.append(a1)
        aList.append(a2)

    matrixA = np.matrix(aList)

    # svd composition
    u, s, v = np.linalg.svd(matrixA)

    # reshape the min singular value into a 3 by 3 matrix
    H = np.reshape(v[8], (3, 3))

    # normalize and now we have h
    H = (1 / H.item(8)) * H
    return H


# Calculate the geometric distance between estimated points and original points
def least_squares(match_pairs, h):
    p1 = np.transpose(np.matrix([match_pairs[0].item(0), match_pairs[0].item(1), 1]))
    estimatep2 = np.dot(h, p1)
    estimatep2 = (1 / estimatep2.item(2)) * estimatep2

    p2 = np.transpose(np.matrix([match_pairs[0].item(2), match_pairs[0].item(3), 1]))
    error = p2 - estimatep2
    return np.linalg.norm(error)


def to_mtx(img):
    H, V, C = img.shape
    mtr = np.zeros((V, H, C), dtype='uint8')
    for i in range(img.shape[0]):
        mtr[:, i] = img[i]

    return mtr


def to_img(mtr):
    V, H, C = mtr.shape
    img = np.zeros((H, V, C), dtype='uint8')
    for i in range(mtr.shape[0]):
        img[:, i] = mtr[i]

    return img


def get_points(im_size):

    points = []
    x_range = np.arange(im_size[1], dtype=int)
    y_range = np.arange(im_size[0], dtype=int)
    y, x = np.meshgrid(y_range, x_range)

    points.append((y.flatten(), x.flatten()))

    return points

def merge_images(cur_image, next_image, H):

    mtr = to_mtx(next_image)
    R, C = (cur_image.shape[1] + next_image.shape[1], cur_image.shape[0])
    dst = np.zeros((R, C, mtr.shape[2]))

    for i in range(mtr.shape[0]):
        for j in range(mtr.shape[1]):
            res = np.dot(H, [i, j, 1])
            i2 = int(res[0, 0] / res[0, 2] + 0.5)
            j2 = int(res[0, 1] / res[0, 2] + 0.5)
            if i2 >= 0 and i2 < R:
                if j2 >= 0 and j2 < C:
                    dst[i2, j2] = mtr[i, j]

    a = to_img(dst)
    # a = cv2.medianBlur(a, 5)
    plt.imshow(a)
    plt.title(''), plt.xticks([]), plt.yticks([])
    plt.show()

    # for i in range(cur_image.shape[0]):
    #     for j in range(cur_image.shape[1]):
    #         if cur_image[i, j][0] != 0 and cur_image[i, j][1] != 0 and cur_image[i, j][2] != 0:
    #             a[i, j] = cur_image[i, j]

    # a[0:cur_image.shape[0], 0:cur_image.shape[1]] = cur_image if cur_image[0] != 0 and cur_image[1] != 0 and
    #             cur_image[2] != 0 else

    # a[0:cur_image.shape[0], 0:cur_image.shape[1]] = cur_image

    non_black_mask = np.all(cur_image != [0, 0, 0], axis=-1)
    cur_image_shape = cur_image.shape
    a_shape = a.shape
    empty_mask = np.zeros((a_shape[0], a_shape[1]), dtype=bool)
    empty_mask[0:cur_image_shape[0], 0:cur_image_shape[1]] = non_black_mask

    a[empty_mask, :] = cur_image[non_black_mask, :]

    # itemindex = np.where(array == item)
    #
    # a[itemindex] = cur_image[itemindex]

    # non_black_mask = np.all(cur_image == [0, 0, 0], axis=-1)


    a = crop_image(a)
    plt.imshow(a)
    plt.show()
    return a


def crop_image(img):
    # Crop top
    if not np.sum(img[0]):
        return crop_image(img[1:])
    # Crop bottom
    elif not np.sum(img[-1]):
        return crop_image(img[:-2])
    # Crop left
    elif not np.sum(img[:, 0]):
        return crop_image(img[:, 1:])
    # Crop right
    elif not np.sum(img[:, -1]):
        return crop_image(img[:, :-2])

    return img


if __name__ == '__main__':
    get_subset_names()
    main()
