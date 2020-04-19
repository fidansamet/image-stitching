import os
import cv2
from matplotlib import pyplot as plt
import numpy as np
import random

dataset_name = "HW2_Dataset"
subsets = {}
orb = cv2.ORB_create()  # Initiate ORB detector
plt.style.use("ggplot")


def get_subset_names():
    root, directories, files = next(os.walk(dataset_name))
    for directory in sorted(directories):
        subsets[directory] = sorted(os.listdir(dataset_name + "/" + directory))


def main():
    for subset_name, subset_images_names in subsets.items():
        feature_points_plot = None
        feature_matching_plot = None
        stitched = cv2.imread(dataset_name + "/" + subset_name + '/' + subset_images_names[0])
        for i in range(0, len(subset_images_names) - 1):
            print(subset_images_names[i])

            # Read current and next images
            next_image = cv2.imread(dataset_name + "/" + subset_name + '/' + subset_images_names[i + 1])
            next_image_gray = cv2.cvtColor(next_image, cv2.COLOR_BGR2GRAY)
            cur_image = cv2.imread(dataset_name + "/" + subset_name + '/' + subset_images_names[i])
            cur_image_gray = cv2.cvtColor(cur_image, cv2.COLOR_BGR2GRAY)

            # Read ground truth panorama image
            ground_truth = cv2.imread(dataset_name + "/" + subset_name + '_gt.png')

            # Feature extraction, feature matching, Homography finding, merging by transformation
            stitch_images(cur_image_gray, next_image_gray, feature_points_plot, feature_matching_plot)

        # result_image = np.concatenate((feature_points_plot, cv2.cvtColor(feature_matching_plot, cv2.COLOR_BGR2RGB)), axis=0)
        # plt.imshow(feature_points_plot)
        # plt.title(''), plt.xticks([]), plt.yticks([])
        # plt.show()


def feature_extraction(img):
    # Feature extraction: find the key points, compute the descriptors with ORB
    key_points, desc = orb.detectAndCompute(img, None)

    # Plots showing feature points for each ordered pair of sub-image
    feature_img = cv2.drawKeypoints(img, key_points, None, color=(0, 255, 0), flags=0)
    plt.imshow(feature_img)
    plt.title(''), plt.xticks([]), plt.yticks([])
    plt.show()

    # if feature_points_plot is None:
    #     feature_points_plot = cur_feature_img
    # else:
    #     feature_points_plot = np.concatenate((feature_points_plot, cv2.cvtColor(next_feature_img, cv2.COLOR_BGR2RGB)),
    #                                          axis=1)

    return key_points, desc


def feature_matching(cur_image, cur_feature_pts, cur_descs, next_image, next_feature_pts, next_descs):
    # each element needs to have 2-nearest neighbors, each list of descriptors needs to have more than 2 elements each
    nearest_neighbor_num = 2

    # Matchers
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    if (cur_descs is not None and len(cur_descs) > nearest_neighbor_num and
            next_descs is not None and len(next_descs) > nearest_neighbor_num):

        # Get knn detector
        matches = bf.knnMatch(next_descs, cur_descs, k=nearest_neighbor_num)

        # Apply ratio test
        good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append([m])

        matches = np.asarray(good)

        # cv2.drawMatchesKnn expects list of lists as matches.
        img3 = cv2.drawMatchesKnn(cur_image, cur_feature_pts, next_image, next_feature_pts, good, None, flags=2)
        plt.imshow(img3)
        plt.show()

        # if feature_matching_plot is None:
        #     feature_matching_plot = macthed
        # else:
        #     feature_matching_plot = np.concatenate((feature_matching_plot, cv2.cvtColor(macthed, cv2.COLOR_BGR2RGB)),
        #                                            axis=1)

        return matches
    else:
        plt.imshow(next_image)
        plt.title(''), plt.xticks([]), plt.yticks([])
        plt.show()
        # if feature_matching_plot is None:
        #     feature_matching_plot = next_image
        # else:
        #     feature_matching_plot = np.concatenate((feature_matching_plot, cv2.cvtColor(next_image, cv2.COLOR_BGR2RGB)),
        #                                            axis=1)
        return None


def stitch_images(cur_image, next_image, feature_points_plot, feature_matching_plot):
    # Feature extraction
    cur_feature_pts, cur_descs = feature_extraction(cur_image)
    next_feature_pts, next_descs = feature_extraction(next_image)

    # Feature matching
    matches = feature_matching(cur_image, cur_feature_pts, cur_descs, next_image, next_feature_pts, next_descs)

    # Find Homography
    if len(matches[:, 0]) >= 4:

        correspondenceList = []

        if matches is not None or matches is not []:
            for match in matches[:, 0]:
                (x1, y1) = next_feature_pts[match.queryIdx].pt
                (x2, y2) = cur_feature_pts[match.trainIdx].pt
                correspondenceList.append([x1, y1, x2, y2])

        print(correspondenceList)
        corrs = np.matrix(correspondenceList)

        # run ransac algorithm
        H, inliers = ransac(corrs, 5.0)
        # H, masked = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        print("Final homography: ", H)
        print("Final inliers count: ", len(inliers))

        dst = cv2.warpPerspective(next_image, H, (cur_image.shape[1] + next_image.shape[1], cur_image.shape[0]))
        dst[0:cur_image.shape[0], 0:cur_image.shape[1]] = cur_image
        plt.imshow(dst)
        plt.show()
        stitched = dst
    else:
        print("Can’t find enough keypoints.")

    return feature_points_plot, feature_matching_plot


def ransac(corr, thresh):
    max_inliers = []
    best_H = None

    for i in range(1000):
        # Find 4 feature (random) points to calculate a homography
        corr1 = corr[random.randrange(0, len(corr))]
        corr2 = corr[random.randrange(0, len(corr))]
        randomFour = np.vstack((corr1, corr2))
        corr3 = corr[random.randrange(0, len(corr))]
        randomFour = np.vstack((randomFour, corr3))
        corr4 = corr[random.randrange(0, len(corr))]
        randomFour = np.vstack((randomFour, corr4))

        # Compute homography
        H = find_homography(randomFour)
        inliers = []

        # Compute inliers where ||pi’, H pi || <ε
        for i in range(len(corr)):
            d = least_squares(corr[i], H)
            if d < 5:
                inliers.append(corr[i])

        # Keep largest set of inliers
        if len(inliers) > len(max_inliers):
            max_inliers = inliers
            best_H = H

        print("Corr size: ", len(corr), " NumInliers: ", len(inliers), "Max inliers: ", len(max_inliers))

        if len(max_inliers) > (len(corr) * thresh):
            break

    return best_H, max_inliers


def find_homography(correspondences):
    # loop through correspondences and create assemble matrix
    aList = []
    for corr in correspondences:
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
def least_squares(correspondence, h):
    p1 = np.transpose(np.matrix([correspondence[0].item(0), correspondence[0].item(1), 1]))
    estimatep2 = np.dot(h, p1)
    estimatep2 = (1 / estimatep2.item(2)) * estimatep2

    p2 = np.transpose(np.matrix([correspondence[0].item(2), correspondence[0].item(3), 1]))
    error = p2 - estimatep2
    return np.linalg.norm(error)


if __name__ == '__main__':
    get_subset_names()
    main()
