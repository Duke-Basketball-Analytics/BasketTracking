import cv2
import numpy as np
import re
import ipdb
import os
import matplotlib.pyplot as plt

from erosion import *
from binarization import *
from keypoint_selection import *

def frame_masking(frame, plot=False):
    gray_frame = binarize_erode_dilate(image, threshold=True, iterations=4)
    blob_frame, contours_court = blob_detection(gray_frame, plot=False)
    court_shape_bin, contour_vertices = rectangularize_court(blob_frame.copy(), contours_court, title = f"OFFENSE-49_OSU: Frame {frame_num}", plot=False)
    if plot:
        isolated_court = cv2.bitwise_and(image, image, mask=court_shape_bin)
        cv2.imshow('Isolated Court', isolated_court)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    blob_frame[court_shape_bin == 0] = 0
    masked_binary = add_border(blob_frame, contour_vertices, court_shape_bin)
    masked_frame = cv2.bitwise_and(frame, frame, mask=masked_binary)
    return masked_frame

def add_border(blob_frame, contour_vertices, court_shape_bin):
    # Add a white border around the court
    outer_mask = np.zeros(blob_frame.shape, dtype=np.uint8)
    cv2.fillPoly(outer_mask, [contour_vertices], 1)
    outer_mask = dilation(outer_mask, k=5)
    inner_mask = np.ones(blob_frame.shape, dtype=np.uint8)
    inner_mask[court_shape_bin == 255] = 0
    inner_mask = dilation(inner_mask, k=10)
    border = outer_mask * inner_mask
    border[border > 0] = 255
    #border = erosion(border, k=2)
    blob_frame = blob_frame + border
    dilated_blob = dilation(blob_frame, k=3, iter=4)
    inverted_blob = 255 - dilated_blob
    clean_blob = block_cleanup(inverted_blob, dilated_blob, plot=True)
    return clean_blob
    

if __name__ == "__main__":
    video_id = "OFFENSE-49_OSU"
    frame_num = 550
    video_path = f"videos/{video_id}.mov"
    video = cv2.VideoCapture(video_path)
    video.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ok, image = video.read()
    if not ok:
        print("Failed to read frame")
        exit()
    image = resize_img(image)
    masked_frame = frame_masking(image)
    plot_image_rectification(video_id=video_id,
                            frame_num=frame_num,
                            image=masked_frame)
    