import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

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


def warp_image(image, H):
    """
    Warps an input image using a homography matrix H. This function handles both grayscale and color images. 
    """
    rows, cols = image.shape[:2]
    warped = cv2.warpPerspective(image, H, (cols, rows))
    return warped

def warp_between_image_planes(A, B, H):
    """
    Warps image A to match image B using homography matrix H. Returns the warped image of the same shape as B.
    """
    # Warp each color channel separately
    channels_A = cv2.split(A)
    warped_channels = []
    for channel in channels_A:
        warped_channel = warp_image(channel, H)
        warped_channels.append(warped_channel)
    warped_A = cv2.merge(warped_channels)
    
    # Compute inverse warp to avoid holes in the output
    rows, cols = B.shape[:2]
    points = np.array([[0, 0], [0, rows-1], [cols-1, rows-1], [cols-1, 0]], dtype=np.float32)
    inverse_H = np.linalg.inv(H)
    warped_points = cv2.perspectiveTransform(points.reshape(1, -1, 2), inverse_H).reshape(-1, 2)
    x_min, y_min = np.int32(warped_points.min(axis=0))
    x_max, y_max = np.int32(warped_points.max(axis=0))
    
    # Sample pixels from the proper coordinates in the source image
    warped_B = np.zeros_like(B)
    for i in range(y_min, y_max):
        for j in range(x_min, x_max):
            point = np.array([[j, i]], dtype=np.float32)
            warped_point = cv2.perspectiveTransform(point.reshape(1, -1, 2), H).reshape(-1, 2)
            x, y = warped_point[0]
            if x < 0 or x >= cols or y < 0 or y >= rows:
                continue
            x1, y1 = int(np.floor(x)), int(np.floor(y))
            x2, y2 = int(np.ceil(x)), int(np.ceil(y))
            if x2 == cols:
                x2 = cols - 1
            if y2 == rows:
                y2 = rows - 1
            alpha = x - x1
            beta = y - y1
            pixel = (1 - alpha)*(1 - beta)*A[y1,x1] + alpha*(1 - beta)*A[y1,x2] + (1 - alpha)*beta*A[y2,x1] + alpha*beta*A[y2,x2]
            if j < 0 or j >= cols or i < 0 or i >= rows:
                continue
            warped_B[i, j] = pixel
    
    return warped_B

# Example usage
# A = cv2.imread('image_A.jpg')
# B = cv2.imread('image_B.jpg')
# H = np.array([[1.2, 0.3, -50], [-0.1, 1.4, 20], [0, 0, 1]])  # Example homography matrix
# warped_A = warp_between_image_planes(A, B, H)
# cv2.imshow('Warped A', warped_A)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# Load ImageA and ImageB
imgA = cv2.imread('uttower1.jpg')
imgB =cv2.imread('uttower2.jpg')

# Recover homography matrix H using feature matching and RANSAC (not shown)
ptA, ptB = correspondences('uttower1.jpg','uttower2.jpg', 8)
H = compute_homography(ptA, ptB)
# Warp ImageA to match ImageB using H
warped_imgA = warp_between_image_planes(imgA, imgB, H)

# Display the warped ImageA
cv2.imshow('Warped ImageA', warped_imgA)
cv2.waitKey(0)
cv2.destroyAllWindows()


