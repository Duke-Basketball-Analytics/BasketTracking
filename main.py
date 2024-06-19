import os
import ipdb

from matplotlib import pyplot as plt

from ball_detect_track import BallDetectTrack
from player import Player
from rectify_court import *
from video_handler import *

BOTTOMCUT = 550
TOPCUT = 100

def binarization(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    adjusted = cv2.equalizeHist(gray)
    lower_thresh=230
    upper_thresh=255
    # Apply double thresholding to create a binary image within the specified range
    # Lower thresholding (all pixels above lower_thresh are set to 255)
    _, lower_result = cv2.threshold(adjusted, lower_thresh, 255, cv2.THRESH_BINARY)
    # Upper thresholding (all pixels above upper_thresh are set to 0)
    _, upper_result = cv2.threshold(adjusted, upper_thresh, 255, cv2.THRESH_BINARY_INV)
    
    # Combine both thresholds to get the final mask
    final_binary = cv2.bitwise_and(lower_result, upper_result)
    return final_binary


def get_frames(video_path, mod):
    frames = []
    cap = cv2.VideoCapture(video_path)
    index = 0
    BOTTOMCUT = 550
    TOPCUT = 100

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame is None:
            cap.release()
            print("Released Video Resource")
            break

        if (index % mod) == 0:
            #frame = binarization(frame)
            frames.append(frame[TOPCUT:BOTTOMCUT, :])

        if cv2.waitKey(20) == ord('q'): break
        index += 1

    cap.release()
    cv2.destroyAllWindows()

    print(f"Number of frames : {len(frames)}")
    # plt.title(f"Centrale {frames[central_frame].shape}")
    # plt.imshow(frames[central_frame])
    # plt.show()

    return frames

def resize_img(img, scale = None):
       # Resize the image
    if scale:
        #ipdb.set_trace()
        h, w, _ = img.shape
        w = int(w*scale)
        h = int(h*scale)
        dimensions = (w,h)
        img = cv2.resize(img, dimensions)
        
    else:
        dimensions = (960, 540)
        img = cv2.resize(img, dimensions)
    return img


#####################################################################
if __name__ == '__main__':
    # COURT REAL SIZES
    # 28m horizontal lines
    # 15m vertical lines
    # loading already computed panoramas
    if os.path.exists('resources/OFFENSE-40_richmond/pano.png'):
        pano = cv2.imread("resources/OFFENSE-40_richmond/pano.png")
    # if os.path.exists('resources/pano.png'):  
    #     pano = cv2.imread("resources/pano.png")
    else:
        #central_frame = 36
        frames = get_frames('resources/debugging_videos/OFFENSE-40_richmond.mov', mod=3)
        frames = frames[::-1]
        central_frame = len(frames) - 45
        frames_flipped = [cv2.flip(frames[i], 1) for i in range(central_frame)]
        current_mosaic1 = collage(frames[central_frame:], direction=1, plot=True)
        #ipdb.set_trace()
        current_mosaic2 = collage(frames_flipped, direction=-1)
        pano = collage([cv2.flip(current_mosaic2, 1)[:, :-10], current_mosaic1])
        cv2.imwrite("resources/OFFENSE-40_richmond/tempcollage_1.png", cv2.flip(current_mosaic2, 1)[:, :-10])
        cv2.imwrite("resources/OFFENSE-40_richmond/collage1_1.png", current_mosaic1)
        cv2.imwrite("resources/OFFENSE-40_richmond/collage2_1.png", current_mosaic2)
        cv2.imwrite("resources/OFFENSE-40_richmond/pano.png", pano)
    
    
    if os.path.exists('resources/OFFENSE-40_richmond/pano_enhanced.png'):
        pano_enhanced = cv2.imread("resources/OFFENSE-40_richmond/pano_enhanced.png")
    # if os.path.exists('resources/pano_enhanced.png'):
    #     pano_enhanced = cv2.imread("resources/pano_enhanced.png")
        plt_plot(pano, save_path=None, title="Panorama")
    else:
        pano_enhanced = pano
        for file in os.listdir("resources/OFFENSE-40_richmond/snapshots/"):
            frame = cv2.imread("resources/OFFENSE-40_richmond/snapshots/" + file)[TOPCUT:BOTTOMCUT, :]
            pano_enhanced = add_frame(frame, pano, pano_enhanced, plot=True, file=file)
        cv2.imwrite("resources/OFFENSE-40_richmond/pano_enhanced.png", pano_enhanced)


    ###################################
    #pano_enhanced = np.vstack((pano_enhanced,
    #                           np.zeros((100, pano_enhanced.shape[1], pano_enhanced.shape[2]), dtype=pano.dtype)))
    # plt_plot(pano_enhanced, title="Pano_enhanced",
    #          save_path = "resources/OFFENSE-40_richmond/pano_enhanced_debug.png")
    
    img = binarize_erode_dilate(pano_enhanced, plot=False)
    simplified_court, corners = (rectangularize_court(img, plot=False))
    simplified_court = 255 - np.uint8(simplified_court)

    indices = np.argsort(corners[:, 0])
    corners = corners[indices]

    # plt_plot(simplified_court, title="Corner Detection", cmap="gray", additional_points=corners, 
    #          save_path = "resources/OFFENSE-40_richmond/simplified_court_corner_detection.png")

    rectified = rectify(pano_enhanced, corners, plot=False)

    plt_plot(rectified, title="Rectified Image",
             save_path = "resources/OFFENSE-40_richmond/rectified_image.png")
    ipdb.set_trace()
    # correspondences map-pano
    map = cv2.imread("resources/2d_map.png")
    # scale = rectified.shape[0] / map.shape[0]
    # map = cv2.resize(map, (int(scale * map.shape[1]), int(scale * map.shape[0])))
    # resized = cv2.resize(rectified, (map.shape[1], map.shape[0]))
    # map = cv2.resize(map, (rectified.shape[1], rectified.shape[0]))

    # My M1 homography matrix maps directly to this size.
    map = cv2.resize(map, (960,540))
    

    cv2.imwrite("resources/OFFENSE-40_richmond/map.png", map)

    video = cv2.VideoCapture("resources/debugging_videos/OFFENSE-40_richmond.mov")

    players = []
    for i in range(1, 6):
        players.append(Player(i, 'blue', hsv2bgr(COLORS['blue'][2])))
        players.append(Player(i, 'white', hsv2bgr(COLORS['white'][2])))
    players.append(Player(0, 'referee', hsv2bgr(COLORS['referee'][2])))

    feet_detector = FeetDetector(players)
    ball_detect_track = BallDetectTrack(players)
    video_handler = VideoHandler(pano_enhanced, video, ball_detect_track, feet_detector, map)

    ipdb.set_trace()
    
    video_handler.run_detectors()
