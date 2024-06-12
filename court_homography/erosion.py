import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import ipdb
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
import seaborn as sns

from plotting import *
from binarization import *

def binarize_erode_dilate(img, iterations = 4, threshold=True, plot=False, save=False):
    # plt.imshow(gray, cmap='gray')
    # plt.show()
    if threshold:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if plot:
            plt.imshow(gray, cmap='gray')
            plt.show()
        # Apply double thresholding to create a binary image within the specified range
        # Lower thresholding (all pixels above lower_thresh are set to 255)
        _, lower_result = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        # Upper thresholding (all pixels above upper_thresh are set to 0)
        _, upper_result = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY_INV)
        
        # Combine both thresholds to get the final mask
        img = cv2.bitwise_and(lower_result, upper_result)
    # plt.imshow(img_otsu, cmap='gray')
    # plt.show()

    # kernel = np.array([[0, 0, 0],
    #                    [1, 1, 1],
    #                    [0, 0, 0]], np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    img_otsu = cv2.erode(img, kernel, iterations=iterations)
    img_otsu = cv2.dilate(img_otsu, kernel, iterations=iterations)
    if plot:
        plt.imshow(img_otsu, cmap='gray')
        plt.show()
    return img_otsu

def blob_detection(frame, plot=False):
     # BLOB FILTERING & BLOB DETECTION

    # adding a little frame to enable detection
    # of blobs that touch the borders
    frame[-4: -1] = frame[0:3] = 0
    frame[:, 0:3] = frame[:, -4:-1] = 0

    mask = np.zeros(frame.shape, dtype=np.uint8)
    cnts = cv2.findContours(frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_court = []

    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    threshold_area = 10000
    for c in cnts:
        area = cv2.contourArea(c)
        if area > threshold_area:
            cv2.drawContours(mask, [c], -1, 255, -1)
            contours_court.append(c)
    # ipdb.set_trace()
    frame = mask
    if plot: plt_plot(frame, title="After Blob Detection", cmap="gray",
                      save_path=None)
    
    return frame, contours_court


def rectangularize_court(frame, contours_court, title, plot=False):

    simple_court = np.zeros(frame.shape)
    if len(contours_court) > 1:
        final_contours = np.vstack(contours_court)
    elif len(contours_court) == 1:
        final_contours = contours_court[0]
    else:
        print("No court detected.")
        return simple_court, None
    # convex hull
    hull = cv2.convexHull(final_contours)
    cv2.drawContours(frame, [hull], 0, 100, 2)
    if plot: plt_plot(frame, title="After ConvexHull", cmap="gray",
                      additional_points=hull.reshape((-1, 2)),
                      save_path=None)
    # ipdb.set_trace()
    # fitting a poly to the hull
    epsilon = 0.01 * cv2.arcLength(hull, True)
    approx = cv2.approxPolyDP(hull, epsilon, True)
    corners = approx.reshape(-1, 2)
    cv2.drawContours(frame, [approx], 0, 100, 5)
    cv2.drawContours(simple_court, [approx], 0, 255, 3)

    if plot:
        plt_plot(frame, title="After Rectangular Fitting", cmap="gray", additional_points=hull.reshape((-1, 2)),
                 save_path=None)
        plt_plot(simple_court, title=f"Rectangularized Court {title}", cmap="gray", additional_points=hull.reshape((-1, 2)),
                 save_path=None)
        print("simplified contour has", len(approx), "points")
    

    isolated_court = np.zeros(simple_court.shape, dtype=np.uint8)
    cv2.fillPoly(isolated_court, [approx], 255)

    return isolated_court, approx

def isolate_court_color(frame):
    #resize image
    # Convert the image from BGR to HSV color space
    hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Reshape the image to be a list of pixels
    pixels = hsv_image.reshape((-1, 3))
    # Perform KMeans clustering
    # kmeans = KMeans(n_clusters=n_clusters, random_state=42)

    ###### Apply DBScan
    clustering = DBSCAN(eps=30, min_samples=600).fit(pixels)

    # Reshape labels to the original image shape
    labels = clustering.labels_
    mask = labels.reshape(hsv_image.shape[0], hsv_image.shape[1])
    
    # Find the largest cluster
    label_histogram = np.bincount(labels[labels >= 0])
    court_cluster_index = label_histogram.argmax()

    # Create a binary mask where the court pixels are white
    court_mask = np.where(mask == court_cluster_index, 255, 0).astype('uint8')

    # Optionally, apply the mask to visualize the result
    result = cv2.bitwise_and(frame, frame, mask=court_mask)

    # Show the result
    cv2.imshow('Court Isolation', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return court_mask

def plot_grayscale_density(image):
    
    # Flatten the image to 1D array for easier handling in plotting
    pixels = image.flatten()
    
    # Plot the density of the grayscale values
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    sns.kdeplot(pixels, bw_adjust=0.5, fill=True, common_norm=True, color='gray')
    plt.title('Density Plot of Grayscale Values')
    plt.xlabel('Grayscale Intensity')
    plt.ylabel('Density')
    plt.xlim([0, 255])  # Grayscale range from 0 to 255
    plt.show()

    
def scale_image(image):
    # Specify the scale factor
    scale_factor = 0.125  # Reducing the dimensions by a factor of 4

    # Calculate the new dimensions
    new_width = int(image.shape[1] * scale_factor)
    new_height = int(image.shape[0] * scale_factor)

    # Resize the image
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    # # Show the resized image
    # cv2.imshow('Resized Image', resized_image)

    # # Save the resized image if necessary
    # cv2.imwrite('path_to_save_resized_image.jpg', resized_image)

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return resized_image

def dilation(img, k = 3, iter = 1):
    # Define the kernel size for dilation
    kernel_size = k
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # Dilate the mask
    mask = img
    dilated_mask = cv2.dilate(mask, kernel, iterations=iter)

    return dilated_mask

def erosion(img, k = 3, iter = 1):
    kernel_size = k
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # Erode the mask
    mask = img
    eroded_mask = cv2.erode(mask, kernel, iterations=iter)

    return 

def block_cleanup(inverted_img, original_img, plot=False):

    contours, hierarchy = cv2.findContours(inverted_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    modified_mask = original_img.copy()
    inverted_mask_with_rectangles = inverted_img.copy()
    largest_c = []
    threshold_area = 10000
    for c in contours:
        area = cv2.contourArea(c)
        if area > threshold_area:
            largest_c.append(c)

    for contour in largest_c:
        # Calculate the bounding rectangle for each contour
        x, y, w, h = cv2.boundingRect(contour)

        # Draw the rectangle on the copied mask (optional, for visualization)
        cv2.rectangle(inverted_mask_with_rectangles, (x-5, y-5), (x+w+5, y+h+5), (255), 1)

        # Fill the rectangle area with zeros in the original mask
        modified_mask[y-5:y+h+5, x-5:x+w+5] = 0
    
    # Display the masks
    if plot:
        cv2.imshow('Original Mask', original_img)
        cv2.imshow('Mask with Rectangles', inverted_mask_with_rectangles)
        cv2.imshow('Modified Mask', modified_mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return modified_mask
    

if __name__ == "__main__":

    # Contour for Duke court
    frame_num = 550
    video = cv2.VideoCapture("videos\OFFENSE-49_OSU.mov")
    video.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ok, frame = video.read()
    if not ok:
        print("Failed to read frame")
    else:
        plt_plot(frame, title=f"Frame Number: {frame_num}")
        gray_frame = binarize_erode_dilate(frame, plot=True)
        blob_frame, contours_court = blob_detection(gray_frame, plot=False)
        rectangle_frame, contour_vertices = rectangularize_court(blob_frame.copy(), contours_court, title = f"OFFENSE-49_OSU: Frame {frame_num}", plot=False)
        # remove everything outside the court
        blob_frame[rectangle_frame == 0] = 0

        # Add a white border around the court
        
        outer_mask = np.zeros(blob_frame.shape, dtype=np.uint8)
        cv2.fillPoly(outer_mask, [contour_vertices], 1)
        outer_mask = dilation(outer_mask, k=5)
        inner_mask = np.ones(blob_frame.shape, dtype=np.uint8)
        inner_mask[rectangle_frame == 255] = 0
        inner_mask = dilation(inner_mask, k=10)
        border = outer_mask * inner_mask
        border[border > 0] = 255
        #border = erosion(border, k=2)
        blob_frame = blob_frame + border
        dilated_blob = dilation(blob_frame, k=3, iter=4)
        inverted_blob = 255 - dilated_blob

        clean_blob = block_cleanup(inverted_blob, dilated_blob, plot=True)

        isolated_court = cv2.bitwise_and(frame, frame, mask=dilated_blob)
        binarized_court = binarize_image(isolated_court, lower_thresh=245, upper_thresh=255, plot=False)
        dilated_court = dilation(binarized_court, k=2, iter=1)
        ipdb.set_trace()
        # plt_plot(dilated_court, title="Line Detection with Obstacles Removed", 
        #          cmap="gray", save_path=None)
        #plt_plot(isolated_court, title="Isolated Court with Obstacles Removed")
        plt.imshow(dilated_court, cmap="gray")
        plt.title("Line Detection with Obstacles Removed")
        plt.show()


    