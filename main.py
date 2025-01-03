import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image


# write code to get manually identified corresponding points from two views
def correspondences(imgA,imgB, nums=10):
    #load image from path
    image_A = Image.open(imgA)
    image_B = Image.open(imgB)
    #create a new figure window for plotting
    figure = plt.figure()

    #plotting image A
    temp_A = figure.add_subplot(1, 2, 1)
    plt.title("Alert:click {num} points".format(num=int(nums / 2)))
    
    #plotting image B
    temp_B = figure.add_subplot(1, 2, 2)
    plt.title("Alert:click {num} points".format(num=int(nums / 2)))
    
    #show picture A and B
    temp_A.imshow(image_A)
    temp_B.imshow(image_B)

    #select points on a plit using the mouse by matplotlib function
    correspondences = plt.ginput(nums, timeout=0)
    correspondences = np.reshape(correspondences, (int(nums / 2), -1))
    ptA = correspondences[:, [0, 1]]
    ptB = correspondences[:, [2, 3]]
    return ptA, ptB

#function that takes a set of corresponding image points and computes the associated 3 × 3 homography matrix H
def compute_homography(ptA, ptB):
    #The Direct Linear Transformation (DLT) algorithm is used to compute the homography matrix.Return the homography matrix.
    
    points = ptA.shape[0]
    # Construct matrix A
    A = np.zeros((2 * points, 9))
    for i in range(points):
        x, y = ptA[i]
        u, v = ptB[i]
        A[2 * i, :] = [-x, -y, -1, 0, 0, 0, x * u, y * u, u]
        A[2 * i + 1, :] = [0, 0, 0, -x, -y, -1, x * v, y * v, v]
    # SVD is used to solve the homography matrix.
    U, S, V = np.linalg.svd(A)
    H = V[-1, :].reshape((3, 3))
    # Normalize H
    H /= H[2, 2]
    return H

#Verify that the homography matrix your function computes is correct by mapping the clicked 
# image points from one view to the other
def verify_homography_matrix(image_A,image_B, H):
    imageA = cv2.imread(image_A)
    imageB = cv2.imread(image_B)

    figure = plt.figure()
    #plotting image A and B
    temp_A = figure.add_subplot(1, 2, 1)
    temp_B = figure.add_subplot(1, 2, 2)
    #show image
    temp_A.imshow(imageA)
    temp_B.imshow(imageB)

    #select points on a plit using the mouse by matplotlib function
    points = plt.ginput(1)
    while True:
        points = plt.ginput(1)
        points = np.reshape(points, (1 * 2, -1))
        matrix_transpose = np.array([points[0], points[1], 1], dtype=object).transpose()
        var = np.dot(H, matrix_transpose).transpose()
        x = var[0] / var[2]
        y = var[1] / var[2]
        temp_A.scatter(points[0], points[1])
        temp_B.scatter([x], [y])



def warpImage(ImageA,ImageB, H):

    #inverse H matrix
    H_inverse = np.linalg.inv(H)

    # recovered homography matrix and an image, and return a new image that is the warp of the input image using H. 
    ImageA_H, ImageA_W, c = ImageA.shape
    ImageB_H, ImageB_W, c = ImageB.shape

    min_x = float("inf")
    min_y = float("inf")
    max_x = float("-inf")
    max_y = float("-inf")
    
    temp_A =  [(0,0), (ImageA_H, ImageA_W), (0, ImageA_W), (ImageA_H, 0)]
    temp_B =  [(0,0), (ImageB_H, ImageB_W), (0, ImageB_W), (ImageB_H, 0)]
    for i,j in temp_A:
        x, y, w = np.matmul(H, [j , i, 1])
        y = y/w
        x = x/w
        if x > max_x:
            max_x = int(x)
        if x < min_x:
            min_x = int(x)
        if y > max_y:
            max_y = int(y)
        if y < min_y:
            min_y = int(y)

    warp_image = np.zeros((max_y - min_y,max_x - min_x, 3))
    # transformed coordinates will typically be sub-pixel values
    for i in range(0, max_x - min_x):
        for j in range (0, max_y - min_y):
            x, y, w = np.matmul(H_inverse, [i + min_x, j + min_y, 1])
            x = int(x/w)
            y = int(y/w)
            a = 0
            b = 0
            c = 0
            if not (y < 0 or y >= ImageA_H or x < 0 or x >= ImageA_W):
                a, b, c = ImageA[y, x, :]
            warp_image[j, i, :] = [a/255, b/255, c/255]

    current_x = min_x
    current_y = min_y
    current_mx = max_x
    current_my = max_y
    for i,j in temp_B:
        if j > max_x:
            max_x = int(j)
        if j < min_x:
            min_x = int(j)
        if i > max_y:
            max_y = int(i)
        if i < min_y:
            min_y = int(i)
    #To avoid holes in the output, use an inverse warp. Warp the points from the source image into
    # the reference frame of the destination, and compute the bounding box in that new reference frame.
    
    merge_image = np.zeros(((max_y - min_y),(max_x - min_x), 3))

    for i in range(min_x, max_x):
        for j in range (min_y, max_y):
            a = 0
            b = 0
            c = 0
            if not (j < current_y or j >= current_my or i < current_x or i >= current_mx):
                a, b, c = warp_image[j - current_y, i - current_x, :]
                if a == 0.0 or b == 0.0 or c == 0.0:
                    if not (j < 0 or j >= ImageB_H or i < 0 or i >= ImageB_W):
                        a, b, c = ImageB[j, i, :]/255
            else:
                if not (j < 0 or j >= ImageB_H or i < 0 or i >= ImageB_W):
                    a, b, c = ImageB[j, i, :]/255
            merge_image[j - min_y, i- min_x, :] = [a, b, c]

    return (warp_image, merge_image)


def getting_correspondences_sift(image1, image2):
    # Convert images to grayscale
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    # Detect SIFT features and compute descriptors
    sift = cv2.SIFT_create(400)

    #Use SIFT to find keypoints and descriptors
    keypoint1, description1 = sift.detectAndCompute(gray1, None)
    keypoint2, description2 = sift.detectAndCompute(gray2, None)
    #Match descriptors using FLANN (Fast Library for Approximate Nearest Neighbors)
    matches = cv2.BFMatcher().knnMatch(description1, description2, k=2)

    #list for point (list)
    key_point_A = []
    key_point_B = []
    good_matches = []
    #Filter matches based on distance ratio
    for mat in matches:
        if mat[0].distance < 0.7 * mat[1].distance:
            good_matches.append([mat[0]])
            image2_index = mat[0].trainIdx
            image1_index = mat[0].queryIdx
            (x1, y1) = keypoint1[image1_index].pt
            (x2, y2) = keypoint2[image2_index].pt
            # Extract corresponding keypoints from good matches
            key_point_A.append((x1, y1))
            key_point_B.append((x2, y2))
    #Draw matches on image
    result_image = cv2.drawMatchesKnn(gray1, keypoint1, gray2, keypoint2,good_matches, flags=2,outImg=None)
    plt.imshow(result_image), plt.show()
    return key_point_A, key_point_B



#main function

imgA = 'uttower1.jpg'
imgB = 'uttower2.jpg'
Condition = input('Choose number 1.Getting Correspondence 2.Verify Homography 3.Wrapping Images 4.Mosaic Images 5.non-RANSAC 6.RANSAC: ')
if(Condition=='1'):
    ptA, ptB = correspondences(imgA, imgB, 8)
elif(Condition =='2'):
    ptA, ptB = correspondences(imgA, imgB, 8)
    H1 = compute_homography(ptA, ptB)
    verify_homography_matrix(imgA,imgB,H1)
elif Condition =='3':
    ptA, ptB = correspondences(imgA, imgB, 8)
    H1 = compute_homography(ptA, ptB)
    imageA = cv2.imread(imgA)
    imageB = cv2.imread(imgB)
    (warpIm, mergeIm)= warpImage(imageA, imageB, H1)
    cv2.imshow('result', warpIm)
    cv2.waitKey()
elif Condition=='4':
    ptA, ptB = correspondences(imgA, imgB, 8)
    H1 = compute_homography(ptA, ptB)
    imageA = cv2.imread(imgA)
    imageB = cv2.imread(imgB)
    (warpIm, mergeIm)= warpImage(imageA, imageB, H1)
    cv2.imshow('result', mergeIm)
    cv2.waitKey()
elif(Condition =='5'):
    img1 = cv2.imread(imgA)
    img2 = cv2.imread(imgB)
    ptA, ptB = getting_correspondences_sift(img1, img2)
    H, status = cv2.findHomography(np.asarray(ptA), np.asarray(ptB), 0, 4.5) 
    (warpIm, mergeIm)= warpImage(img1, img2, H)
    cv2.imshow('result', mergeIm)
    cv2.waitKey()
elif (Condition =='6'):
    img1 = cv2.imread(imgA)
    img2 = cv2.imread(imgB)
    ptA, ptB = getting_correspondences_sift(img1, img2)
    H, status = cv2.findHomography(np.asarray(ptA), np.asarray(ptB), cv2.RANSAC, 4.5) 
    (warpIm, mergeIm)= warpImage(img1, img2, H)
    cv2.imshow('result', mergeIm)
    cv2.waitKey()
