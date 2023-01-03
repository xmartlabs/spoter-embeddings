import os.path as op
from itertools import chain
from collections import namedtuple
import argparse
import glob

import os

import cv2
import numpy as np
import mediapipe as mp
from tqdm.auto import tqdm

# Import drawing_utils and drawing_styles.
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic
mp_pose = mp.solutions.pose

LEN_LANDMARKS_POSE = len(mp_holistic.PoseLandmark)
LEN_LANDMARKS_HAND = len(mp_holistic.HandLandmark)
TOTAL_LANDMARKS = LEN_LANDMARKS_POSE + 2 * LEN_LANDMARKS_HAND

Landmark = namedtuple("Landmark", ["x", "y"])


class LandmarksResults:
    """
    Wrapper for landmarks results. When not available it fills with 0
    """

    def __init__(
        self,
        results,
        num_landmarks_pose=LEN_LANDMARKS_POSE,
        num_landmarks_hand=LEN_LANDMARKS_HAND,
    ):
        self.results = results
        self.num_landmarks_pose = num_landmarks_pose
        self.num_landmarks_hand = num_landmarks_hand

    @property
    def pose_landmarks(self):
        if self.results.pose_landmarks is None:
            return [Landmark(0, 0)] * self.num_landmarks_pose
        else:
            return self.results.pose_landmarks.landmark

    @property
    def left_hand_landmarks(self):
        if self.results.left_hand_landmarks is None:
            return [Landmark(0, 0)] * self.num_landmarks_hand
        else:
            return self.results.left_hand_landmarks.landmark

    @property
    def right_hand_landmarks(self):
        if self.results.right_hand_landmarks is None:
            return [Landmark(0, 0)] * self.num_landmarks_hand
        else:
            return self.results.right_hand_landmarks.landmark


def get_landmarks(image_orig, holistic, debug=False):
    """
    Runs landmarks detection for single image
    Returns: list of landmarks
    """
    # Convert the BGR image to RGB before processing.
    image = cv2.cvtColor(image_orig, cv2.COLOR_BGR2RGB)
    results = LandmarksResults(holistic.process(image))
    if debug:
        lmks_pose = []
        for lmk in results.pose_landmarks:
            lmks_pose.append(lmk.x)
            lmks_pose.append(lmk.y)
        assert len(lmks_pose) == LEN_LANDMARKS_POSE

        lmks_left_hand = []

        for lmk in results.left_hand_landmarks:
            lmks_left_hand.append(lmk.x)
            lmks_left_hand.append(lmk.y)

        assert (
            len(lmks_left_hand) == 2 * LEN_LANDMARKS_HAND
        ), f"{len(lmks_left_hand)} != {2 * LEN_LANDMARKS_HAND}"

        lmks_right_hand = []

        for lmk in results.right_hand_landmarks:
            lmks_right_hand.append(lmk.x)
            lmks_right_hand.append(lmk.y),

        assert (
            len(lmks_right_hand) == 2 * LEN_LANDMARKS_HAND
        ), f"{len(lmks_right_hand)} != {2 * LEN_LANDMARKS_HAND}"
    landmarks = []
    for lmk in chain(
        results.pose_landmarks,
        results.left_hand_landmarks,
        results.right_hand_landmarks,
    ):
        landmarks.append(lmk.x)
        landmarks.append(lmk.y)
    assert (
        len(landmarks) == TOTAL_LANDMARKS * 2
    ), f"{len(landmarks)} != {TOTAL_LANDMARKS * 2}"
    return landmarks


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--videos-folder",
        "-videos",
        help="Path of folder with videos to extract landmarks from",
        required=True,
    )
    parser.add_argument(
        "--output-landmarks",
        "-lmks",
        help="Path of output folder where landmarks npy files will be saved",
        required=True,
    )
    args = parser.parse_args()
    landmarks_output = args.output_landmarks
    videos_folder = args.videos_folder
    os.makedirs(landmarks_output, exist_ok=True)
    for fn_video in tqdm(sorted(glob.glob(op.join(videos_folder, "*mp4")))):
        cap = cv2.VideoCapture(fn_video)
        ret, image_orig = cap.read()
        height, width = image_orig.shape[:2]
        landmarks_video = []
        with tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))) as pbar:
            with mp_holistic.Holistic(
                static_image_mode=False,
                min_detection_confidence=0.5,
                model_complexity=2,
            ) as holistic:
                while ret:
                    try:
                        landmarks = get_landmarks(image_orig, holistic)
                    except Exception as e:
                        print(e)
                        landmarks = get_landmarks(image_orig, holistic, debug=True)
                    ret, image_orig = cap.read()
                    landmarks_video.append(landmarks)
                    pbar.update(1)
        landmarks_video = np.vstack(landmarks_video)
        np.save(
            op.join(landmarks_output, op.basename(fn_video).split(".")[0]),
            landmarks_video,
        )
