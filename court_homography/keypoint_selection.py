import cv2
import matplotlib.pyplot as plt
import ipdb
import numpy as np
import re
from .erosion import *

# Button parameters
button_position = (50, 50)  # Position of the top-left corner of the button
button_size = (100, 50)  # Size of the button (width, height)
button_color = (200, 200, 200)  # Button color
text_color = (0, 0, 0)  # Text color

# Initialization
# image = None
# keypoint_data = None

# Function to display the image and capture clicks
def click_event(event, x, y, flags, param):
    img, points = param['img'], param['points']
    if event == cv2.EVENT_LBUTTONDOWN:
        
        cv2.circle(img, (x, y), 5, (255, 0, 0), -1)
        id = input("Enter the label number for this keypoint")
        points.append((id, x, y))
        cv2.imshow('image', img)

def keypoints(img):

    points = []

    cv2.imshow('image', img)
    #cv2.setMouseCallback('image', click_event, params={'img': img, 'points': points})
    cv2.setMouseCallback('image', click_event, param={'img': img, 'points': points})

    # Wait until 'q' is pressed to quit and print points
    print("Click to select points. Press 'q' to quit and print the selected points.")
    while True:
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    print("Selected Points:", points)

    return points

def plot_keypoints(func):
    def create_plot(image, keypoint_data, title, *args, **kwargs):
        # Step 3: Plot the keypoints on the image
        plt.figure(figsize=(8, 10))
        plt.imshow(image)

        # Iterate over the dictionary and plot each keypoint
        for id, coords in keypoint_data.items():
            x, y = coords
            plt.plot(x, y, 'bo')  # Blue circle at the keypoint location
            plt.text(x, y, id, color='red', fontsize=12)  # Annotate the keypoint with its ID
        plt.title(f"{title}")
        return func(image, keypoint_data, title, *args, **kwargs)
    return create_plot

@plot_keypoints
def save_plot(image, keypoint_data, title):
    pattern = r"[:/.\s\\]"
    sections = re.split(pattern, title)
    sections = [x for x in sections if x]
    print(sections[1])
    ipdb.set_trace()
    if sections[1] == "videos":
        id = sections[2]
        frame_num = sections[5]
        output_path = f"keypoint_images/{id}_F{frame_num}_kpi.png"
    else:
        filename = input("Image filename (include file extension .png or .jpg): ")
        output_path = "keypoint_images/" + filename
    # Save the image
    try:
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1)  # Adjust padding as needed
        print(f"Image saved as {output_path}")
    except Exception as e:
        print(f"Could not save image: {e}")

@plot_keypoints
def show_plot(image, keypoint_data, title):
    plt.show()

def resize_img(img):
       # Resize the image
    dimensions = (960, 540)
    img = cv2.resize(img, dimensions)
    return img

def identify_keypoints(path, frame_num=None):
    if path[-3:] in ["mov", "mp4"]:
        # Read video frame 
        frame_num = frame_num
        video = cv2.VideoCapture(path)
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ok, img = video.read()
        pattern = r"[\\.]"
        vid_id = re.split(pattern, path)[1]
        if not ok:
            print("Failed to read frame")
            return None
    elif path[-3:] in ["jpg", "png"]:
        img = cv2.imread(path)
    else:
        print(f"File extension {path[-3:]} not accepted.")
        return None
    # Resize the image
    img = resize_img(img)

    # Mark keypoints

    keypts = keypoints(img)

    #Convert keypoints to dictionary
    kp_dict = {}
    for id, X, Y in keypts:
        kp_dict[id] = np.array([X, Y])

    # Save keypoints
    save = input("Save Data?")
    if save == 'y':
        att = 0
        while att < 3:
            if vid_id and frame_num:
                file_path = f"keypoints/{vid_id}_F{frame_num}_kp.npz"
            else:
                filename = input("Enter filename ending with '.npz'")
                file_path = "keypoints/" + filename
            try:
                np.savez(file_path, **kp_dict)
                break  # Exit loop on successful save
            except Exception as e:
                print(f"Attempt to save did not work: {e}")
            att += 1
        else:
            print("Exceeded save attempts")
            return None
    return "OK"

def check_keypoints(kp_path, img_path=None, video_path=None, frame_num=None):
    # Read template court
    if img_path:
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image = resize_img(img)
        title = f"Keypoints: {img_path}"
    elif video_path:
        frame_num = frame_num
        video = cv2.VideoCapture(video_path)
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ok, img = video.read()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image = resize_img(img)
        title = f"Keypoints: {video_path}, Frame:{frame_num}"
        if not ok:
            print("Failed to read frame")
            return "Fail"

    keypoint_data = np.load(kp_path)
    show_plot(image, keypoint_data, title=title)
    if input("Save plot?") == 'y':
        save_plot(image, keypoint_data, title=title)
        return "OK"
    return "OK"

def compute_homography(keypoints1, keypoints2):
    if type(keypoints1) == str:
        keypoints_image1 = npz_extraction(keypoints1)
        keypoints_image2 = npz_extraction(keypoints2)
    else:
        keypoints_image1 = keypoints1
        keypoints_image2 = keypoints2

    # Arrays to hold the matching points
    points1 = []
    points2 = []

    # Extract matching keypoints
    for keyID in keypoints_image1:
        if keyID in keypoints_image2:
            points1.append(keypoints_image1[keyID])
            points2.append(keypoints_image2[keyID])

    # Convert lists of points to numpy arrays of type float32
    points1 = np.array(points1, dtype=np.float32)
    points2 = np.array(points2, dtype=np.float32)

    # Compute the homography matrix using RANSAC
    H, status = cv2.findHomography(points1, points2, 0)

    return H, points1, points2

def npz_extraction(npz_file):
    data = np.load(npz_file)
    npz_dict = {key:data[key] for key in data}
    return npz_dict

def plot_image_rectification(video_id=None, frame_num=None, image=None):
    if image is None:
        video_path = f"videos\{video_id}.mov"
        frame_num = frame_num
        video = cv2.VideoCapture(video_path)
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ok, image = video.read()
        if not ok:
            print("Failed to read frame")
    image = resize_img(image)
    
    # Use the function
    keypoints1 = f"keypoints/{video_id}_F{frame_num}_kp.npz"
    keypoints2 = "keypoints/template_court_kp.npz"
    M, points1, points2 = compute_homography(keypoints1, keypoints2)

    # Define the size of the output image (width, height)
    output_size = (image.shape[1], image.shape[0])  # Same size as original
    # Apply the homography
    transformed_image = cv2.warpPerspective(image, M, output_size)

    image = resize_img(image)
    transformed_image = resize_img(transformed_image)
    # Display the original and transformed image
    cv2.imshow('Original Image', image)
    cv2.imshow('Transformed Image', transformed_image)
    cv2.waitKey(0)
    if input("Save transformation?") == "y":
        cv2.imwrite(f"transformed_images/{video_id}/F{frame_num}.png", transformed_image)
    cv2.destroyAllWindows()




if __name__ == "__main__":
    ### Plot keypoints on top of original image/frame
    # status = check_keypoints(kp_path="keypoints/template_court_kp.npz",
    #                 img_path="court_images/map.png",
    #                 video_path=None,
    #                 frame_num=None)

    ### Calculate homography matrix and plot the transformed image
    plot_image_rectification(video_id="OFFENSE-49_OSU",
                             frame_num=410,
                             image=None)
    exit()
    ### Save keypoints on a new image/frame
    status = identify_keypoints(path="videos\OFFENSE-49_OSU.mov", 
                                frame_num=410)
    
    ### Isolate court before applying transformation
    video_path = f"videos\OFFENSE-49_OSU.mov"
    frame_num = 410
    video = cv2.VideoCapture(video_path)
    video.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ok, image = video.read()
    if not ok:
        print("Failed to read frame")
        exit()
    image = resize_img(image)
    gray_frame = binarize_erode_dilate(image, threshold=True, iterations=4)
    blob_frame, contours_court = blob_detection(gray_frame, plot=False)
    court_shape_bin, contour_vertices = rectangularize_court(blob_frame.copy(), contours_court, title = f"OFFENSE-49_OSU: Frame {frame_num}", plot=False)
    isolated_court = cv2.bitwise_and(image, image, mask=court_shape_bin)

    cv2.imshow('Isolated Court', isolated_court)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    ipdb.set_trace()
    