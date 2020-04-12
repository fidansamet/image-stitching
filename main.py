import os
import cv2
from matplotlib import pyplot as plt

dataset_name = "HW2_Dataset"
dataset_contents = os.listdir(dataset_name)


def main():
    for content in dataset_contents:
        # If it is a directory get images
        if os.path.isdir(dataset_name + "/" + content):
            print(content)
            subset_images = os.listdir(dataset_name + "/" + content)
            for image_name in subset_images:
                print(image_name)
                image = cv2.imread(dataset_name + "/" + content + '/' + image_name)

                # Initiate STAR detector
                orb = cv2.ORB_create()

                # find the keypoints with ORB
                kp = orb.detect(image, None)

                # compute the descriptors with ORB
                kp, des = orb.compute(image, kp)

                # draw only keypoints location,not size and orientation
                img2 = image
                cv2.drawKeypoints(image, kp, img2, color=(0, 255, 0), flags=0)
                plt.imshow(img2), plt.show()


if __name__ == '__main__':
    main()
