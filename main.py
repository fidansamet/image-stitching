import os
import cv2
from matplotlib import pyplot as plt

dataset_name = "HW2_Dataset"
subsets = {}


def get_subset_names():
    root, directories, files = next(os.walk(dataset_name))
    for directory in sorted(directories):
        subsets[directory] = sorted(os.listdir(dataset_name + "/" + directory))


def main():
    for subset_name, subset_images_names in subsets.items():
        for image_name in subset_images_names:
            print(image_name)

            image = cv2.imread(dataset_name + "/" + subset_name + '/' + image_name)

            # Initiate STAR detector
            orb = cv2.ORB_create()

            # find the keypoints with ORB
            key_points = orb.detect(image, None)

            # compute the descriptors with ORB
            key_points, des = orb.compute(image, key_points)

            # draw only keypoints location,not size and orientation
            result_image = cv2.drawKeypoints(image, key_points, None, color=(0, 255, 0), flags=0)

            plt.imshow(result_image)
            plt.show()


if __name__ == '__main__':
    get_subset_names()
    main()
