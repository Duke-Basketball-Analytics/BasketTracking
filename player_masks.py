import os
import ipdb
import cv2

from matplotlib import pyplot as plt

from ball_detect_track import BallDetectTrack
from player import Player
from rectify_court import *
from video_handler import *

def resize_img(img):
       # Resize the image
    dimensions = (960, 540)
    img = cv2.resize(img, dimensions)
    return img

def setup():
    video = cv2.VideoCapture("resources/debugging_videos/OFFENSE-40_richmond.mov")
    pano_enhanced = None

    players = []
    for i in range(1, 6):
        players.append(Player(i, 'green', hsv2bgr(COLORS['green'][2])))
        players.append(Player(i, 'white', hsv2bgr(COLORS['white'][2])))
    players.append(Player(0, 'referee', hsv2bgr(COLORS['referee'][2])))

    feet_detector = FeetDetector(players)
    ball_detect_track = BallDetectTrack(players)
    video_handler = VideoHandler(pano_enhanced, video, ball_detect_track, feet_detector, map)
    
    return video_handler, feet_detector, ball_detect_track


if __name__ == "__main__":
    video_handler, feet_detector, ball_tracker = setup()
    M = M1 = time_index = map_2d = None

    frame_num = 230
    video_id = "OFFENSE-29_richmond"

    video = cv2.VideoCapture(f"court_homography/videos/{video_id}.mov")
    video.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ok, frame = video.read()
    if not ok:
        print("Unable to read frame.")
        exit()
    
    frame = resize_img(frame)

    masks = feet_detector.get_players_pos(M, M1, frame, time_index, map_2d, only_mask=True)

    final_mask = np.zeros((masks[0].shape))
    for mask in masks:
        final_mask += mask
        final_mask = np.where(final_mask < 1, 0, 255)

    final_mask = final_mask.astype(np.uint8)
    kernel_size = 3
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated_mask = cv2.dilate(final_mask, kernel, iterations=5)

    np.save(f"court_homography/player_masks_arrays/{video_id}_F{frame_num}_playermasks.npy", dilated_mask)
    
    # ipdb.set_trace()