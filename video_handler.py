from player_detection import *
from ball_detect_track import *

import skvideo.io

TOPCUT = 0

FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)


class VideoHandler:
    def __init__(self, pano, video, ball_detector, feet_detector, map_2d):
        # Anytime we use M1 we need to use Ms first with destination size == scale*pano dimensions
        #self.Ms = np.load("resources/OFFENSE-40_richmond/scaling_matrix_033.npy")
        # OSU_Pano_Rectify1 has output dimensions (960,540)
        #self.M1 = np.load("resources/OFFENSE-40_richmond/OSU_Pano_Rectify1.npy")
        self.M1 = np.load("Rectify1.npy")
        self.sift = cv2.xfeatures2d.SIFT_create()
        self.pano = pano
        self.video = video
        self.kp1, self.des1 = self.sift.compute(pano, self.sift.detect(pano))
        self.feet_detector = feet_detector
        self.ball_detector = ball_detector
        self.map_2d = map_2d

    def run_detectors(self):
        writer = skvideo.io.FFmpegWriter("resources/debugging_videos/test2.mp4")
        time_index = 0
        homography_matrices = []
        while self.video.isOpened():
            ok, frame = self.video.read()
            if not ok:
                break
            else:
                if 0 <= time_index <= 10:

                    print("\r Computing DEMO: " + str(int(100 * time_index / 10)) + "%",
                          flush=True, end='')

                    frame = frame[TOPCUT:, :]
                    M = self.get_homography(frame, self.des1, self.kp1)
                    homography_matrices.append(M)
                    frame, self.map_2d, map_2d_text = self.feet_detector.get_players_pos(M, self.M1, frame, time_index,
                                                                                         self.map_2d)
                    frame, ball_map_2d = self.ball_detector.ball_tracker(M, self.M1, frame, self.map_2d.copy(),
                                                                         map_2d_text, time_index)
                    vis = np.vstack((frame, cv2.resize(map_2d_text, (frame.shape[1], frame.shape[1] // 2))))

                    #cv2.imshow("Tracking", vis)
                    if time_index % 1 == 0:
                        plt_plot(vis, save_path=f"resources/OFFENSE-40_richmond/demo_frames/frame{time_index}.png")
                    writer.writeFrame(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))

                    k = cv2.waitKey(1) & 0xff
                    if k == 27:
                        break
            time_index += 1
        ipdb.set_trace()
        self.video.release()
        writer.close()
        cv2.destroyAllWindows()

    def get_homography(self, frame, des1, kp1):
        kp2 = self.sift.detect(frame)
        kp2, des2 = self.sift.compute(frame, kp2)
        matches = flann.knnMatch(des1, des2, k=2)
        good = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good.append(m)

        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
        return M

