import os
import cv2
from matplotlib import pyplot as plt
import numpy as np
import random

dataset_name = "HW2_Dataset"
subsets = {}
orb = cv2.ORB_create()  # Initiate STAR detector
plt.style.use("ggplot")


def drawMatches(img1, kp1, img2, kp2, matches, inliers = None):
    # Create a new output image that concatenates the two images together
    rows1 = img1.shape[0]
    cols1 = img1.shape[1]
    rows2 = img2.shape[0]
    cols2 = img2.shape[1]

    out = np.zeros((max([rows1,rows2]),cols1+cols2,3), dtype='uint8')

    # Place the first image to the left
    out[:rows1,:cols1,:] = np.dstack([img1, img1, img1])

    # Place the next image to the right of it
    out[:rows2,cols1:cols1+cols2,:] = np.dstack([img2, img2, img2])

    # For each pair of points we have between both images
    # draw circles, then connect a line between them
    for mat in matches:

        # Get the matching keypoints for each of the images
        img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx

        # x - columns, y - rows
        (x1,y1) = kp1[img1_idx].pt
        (x2,y2) = kp2[img2_idx].pt

        inlier = False

        if inliers is not None:
            for i in inliers:
                if i.item(0) == x1 and i.item(1) == y1 and i.item(2) == x2 and i.item(3) == y2:
                    inlier = True

        # Draw a small circle at both co-ordinates
        cv2.circle(out, (int(x1),int(y1)), 4, (255, 0, 0), 1)
        cv2.circle(out, (int(x2)+cols1,int(y2)), 4, (255, 0, 0), 1)

        # Draw a line in between the two points, draw inliers if we have them
        if inliers is not None and inlier:
            cv2.line(out, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), (0, 255, 0), 1)
        elif inliers is not None:
            cv2.line(out, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), (0, 0, 255), 1)

        if inliers is None:
            cv2.line(out, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), (255, 0, 0), 1)

    return out

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
            # find_homography()

        #result_image = np.concatenate((feature_points_plot, cv2.cvtColor(feature_matching_plot, cv2.COLOR_BGR2RGB)), axis=0)
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


def feature_matching(cur_image, cur_key_points, cur_desc, next_image, next_key_points, next_desc):
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

        macthed = cv2.drawMatchesKnn(cur_image, cur_key_points, next_image, next_key_points, matches, None,
                                     **draw_params)
        plt.imshow(macthed)
        plt.title(''), plt.xticks([]), plt.yticks([])
        plt.show()

        # As per Lowe's ratio test to filter good matches
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)



        # if feature_matching_plot is None:
        #     feature_matching_plot = macthed
        # else:
        #     feature_matching_plot = np.concatenate((feature_matching_plot, cv2.cvtColor(macthed, cv2.COLOR_BGR2RGB)),
        #                                            axis=1)

        # matcher = cv2.BFMatcher(cv2.NORM_L2, True)
        # matches = matcher.match(cur_desc, next_desc)
        # matchImg = cv2.drawMatches(cur_image, cur_key_points, next_image, next_key_points, matches, None, flags=2)
        # cv2.imwrite('Matches.png', matchImg)
        return good_matches
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



def feature_extraction_and_matching(cur_image, next_image, feature_points_plot, feature_matching_plot):

    cur_key_points, cur_desc = feature_extraction(cur_image)
    next_key_points, next_desc = feature_extraction(next_image)

    # Feature matching
    matches = feature_matching(cur_image, cur_key_points, cur_desc, next_image, next_key_points, next_desc)

    correspondenceList = []

    if matches is not None or matches is not []:
        for match in matches:
            (x1, y1) = cur_key_points[match.queryIdx].pt
            (x2, y2) = next_key_points[match.trainIdx].pt
            correspondenceList.append([x1, y1, x2, y2])

    print(correspondenceList)
    corrs = np.matrix(correspondenceList)

    # run ransac algorithm
    finalH, inliers = ransac(corrs, 0.60)      # TODO
    print("Final homography: ", finalH)
    print("Final inliers count: ", len(inliers))

    matchImg = drawMatches(cur_image, cur_key_points, next_image, next_key_points, matches, inliers)
    plt.imshow(matchImg)
    plt.title(''), plt.xticks([]), plt.yticks([])
    plt.show()

    return feature_points_plot, feature_matching_plot


def ransac(corr, thresh):
    maxInliers = []
    finalH = None
    for i in range(1000):
        #find 4 random points to calculate a homography
        corr1 = corr[random.randrange(0, len(corr))]
        corr2 = corr[random.randrange(0, len(corr))]
        randomFour = np.vstack((corr1, corr2))
        corr3 = corr[random.randrange(0, len(corr))]
        randomFour = np.vstack((randomFour, corr3))
        corr4 = corr[random.randrange(0, len(corr))]
        randomFour = np.vstack((randomFour, corr4))

        #call the homography function on those points
        h = find_homography(randomFour)
        inliers = []

        for i in range(len(corr)):
            d = geometricDistance(corr[i], h)
            if d < 5:
                inliers.append(corr[i])

        if len(inliers) > len(maxInliers):
            maxInliers = inliers
            finalH = h
        print("Corr size: ", len(corr), " NumInliers: ", len(inliers), "Max inliers: ", len(maxInliers))

        if len(maxInliers) > (len(corr)*thresh):
            break
    return finalH, maxInliers



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
    h = np.reshape(v[8], (3, 3))

    # normalize and now we have h
    h = (1 / h.item(8)) * h
    return h

    #
    # Calculate the geometric distance between estimated points and original points
    #


def geometricDistance(correspondence, h):
    p1 = np.transpose(np.matrix([correspondence[0].item(0), correspondence[0].item(1), 1]))
    estimatep2 = np.dot(h, p1)
    estimatep2 = (1 / estimatep2.item(2)) * estimatep2

    p2 = np.transpose(np.matrix([correspondence[0].item(2), correspondence[0].item(3), 1]))
    error = p2 - estimatep2
    return np.linalg.norm(error)


if __name__ == '__main__':
    get_subset_names()
    main()
