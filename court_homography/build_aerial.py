import cv2
import numpy as np
import re
import ipdb
import os
import matplotlib.pyplot as plt

from erosion import *
from keypoint_selection import *
from frame_transformation import *
from plotting import *

FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

def merge_frames(frame1, frame2, plot=False):
    # Convert to grayscale
    # Trim bottom off to account for scores box - automate later
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)[:300,:]
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)[:300,:]

    gray1 = cv2.equalizeHist(gray1)
    gray2 = cv2.equalizeHist(gray2)

    sift = cv2.SIFT_create()  # sift instance

    # FINDING FEATURES
    kp1 = sift.detect(gray1)
    kp1, des1 = sift.compute(gray1, kp1)
    kp2 = sift.detect(gray2)
    kp2, des2 = sift.compute(gray2, kp2)

    matches = flann.knnMatch(des1, des2, k=2)

    # Need to draw only good matches, so create a mask
    matchesMask = [[0,0] for i in range(len(matches))]
    good = []
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.8 * n.distance:
            matchesMask[i]=[1,0]
            good.append(m)
    print(f"Number of good correspondences: {len(good)}")    
    
    draw_params = dict(matchColor = (0,255,0),
                       singlePointColor = (255,0,0),
                       matchesMask = matchesMask,
                       flags = cv2.DrawMatchesFlags_DEFAULT)
    
    img3 = cv2.drawMatchesKnn(frame1,kp1,frame2,kp2,matches,None,**draw_params)
    
    plt_plot(img3, title="Keypoint Matching Between Aerial Transformations")

    if len(good) < 70:
        print("Not enough matches found")
        return None

    # Finding an homography
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)

    result = cv2.warpPerspective(frame2,
                                 M,
                                 (frame1.shape[1],
                                  frame1.shape[0]))

    if plot: plt_plot(result, title="Warped new image")

    

if __name__ == "__main__":
    frame1 = cv2.imread("transformed_images/OFFENSE-49_OSU/F410_nomask.png")
    frame2 = cv2.imread("transformed_images/OFFENSE-49_OSU/F550_nomask.png")

    merged_frame = merge_frames(frame1, frame2, plot=True)
