This program stitches sub-images of subset to obtain a panorama image. It reads all subsets of directory with given name and tries to stitch sub-images of subsets to obtain panorama. It needs 4.1.1.26 version OpenCV package and 3.4.2.17 version OpenCV contrib module to run. It reads inputs from DATASET_NAME named directory. In this directory, there must be directories of sub-images of panorama. It reads sub-images and merges them. It also plots ground truth panorama for comparision. Ground truth panorama image must be in DATASET_NAME named directory and must be named as subset name + "_gt".

Outputs will be shown as plots.

To run the program run the following command:

pip3 uninstall opencv-python
pip3 --no-cache-dir install opencv-python==4.1.1.26
pip3 install --no-cache-dir opencv-contrib-python==3.4.2.17
python3 main.py



Functions:

get_subset_names:
This function initializes the "subsets" dictionary which stores directory name as a key and its files as a value in "dataset_name" named directory. "subsets" dictionary's keys and values are sorted since panorama images are formed with continous sub-images. So image names must be in order with panorama sub-images.
	
main:
This function loops over "subsets" dictionary's keys and values two times. In these loops, iterator traverses over current key (a directory) and its values (dictionary's images), and assigns first image of the directory as the current panorama since it is the first sub-image of the panorama.
At the beginning of the first inner loop, for each current and next two sub-images of the panorama, converts to gray scale and then applies $3x3$ gaussian blur filter to reduce the noise in order to extract better features from them. Then, it calls image stitcher function for current and next images. This called function returns a homography matrix if there is enough keypoints and matches. Then it appends this homography matrix to homographies list. If called function does not return a homography matrix, exits the first inner loop and enters the second inner loop.
In the second inner loop, it calculates homography matrix for mathematical approach of panorama image stitching. Then merges so far obtained panorama image and next image by using this calculated matrix. At each iteration it updates panorama image with merged image.
At the end of the second inner loop, it plots final obtained panorama image and given ground truth panorama image.
	
stitch_images:
This function firstly extracts features by calling feature extraction method for two consecutive sub-images. Then with obtained keypoints and descriptors, it calls feature matching method. After obtaining matches, if there is enough number of matches, it calls RANSAC method to find homography matrix with matched pairs and returns calculated matrix. If there is not enough number of matches, it returns none.
	
feature_extraction:
This function extracts features of given image. It obtains keypoints and descriptors by using OpenCV's SURF(Speeded-Up Robust Features) detectAndCompute method. After that it plots detected keypoints.

feature_matching:
This function matches features of 2 consecutive sub-images. Firstly checks whether there is enough number of descriptors or not. If not, returns none as matches. If there is, then performs KNN match of OpenCV’s by using FLANN matcher. After obtaining matches, applies ratio test to eliminate bad matches. Then plots matched lines and returns those matches.

RANSAC:
This function performs RANSAC algorithm to find best homography matrix. It iterates given number of times, and at each iteration it takes 4 random matched pairs, calls homography finder method with those values. After obtaining current homography matrix it calculates number of inliers for this matrix. It calls geometric distance calculator at this point. Then compares this distance with inlier threshold to see if it is inlier or not. After calculating inlier number, updates maximum inlier number and best homography matrix according to current inlier number. Finally if maximum inlier number exceeds threshold it exits for loop. Finally returns best homography matrix found.
	
merge_images:
This function merges given two images according to given homography matrix. It traverses every pixel of next image and calculates its new position in current image by multipying indices with homography matrix. After calculating transformed next image, it copies non-black pixels of current image to that image. It is because black regions left after merge and it overlaps next image’s pixels. Finally returns merged image.

find_homography:
This function calculates homography matrix with given 4 random matched points by using homography estimation.
	
get_geometric_distance:
This function calculates geometric distance between matched points and current points.
	
crop_image:
This function crops 0-sum top, bottom, left and right regions to remove balck regoins in image. It is used to crop unnecessary black regions in merged image.
	
transpose_copy:
This function copies given matrix to new matrix by transposing first and second dimensions. It assigns new matrix type as requested.



Globals:

DATASET_NAME: Specifies the directory name of directories of sub-images.
SURF: SURF key point descriptor.
NEAREST_NEIGHBOR_NUM: Specifies nearest neighbor number for KNN match method.
RANSAC_THRESH: Specifies the threshold where RANSAC algorithm must stop.
INLIER_THRESH: Specifies the inlier distance.
RANSAC_ITERATION: Specifies the number of iteration RANSAC algorithm runs.
subsets: It is a dictionary and specifies subsets and their sub-images in given directory.
index_params: Specifies the index parameters for FLANN based matcher.
search_params: Specifies the search parameters for FLANN based matcher.
brute_force_matcher: It is Brute Force matcher. It is not used in the program right now.
flann_matcher: It is FLANN based matcher.


