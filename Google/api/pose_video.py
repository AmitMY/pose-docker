import json
import imageio
import sys
import numpy as np

sys.path.append("/api")
from pose_util import hand_from_raw

sys.path.append("/hand_tracking")
from hand_tracker import HandTracker

palm_model_path = "/hand_tracking/models/palm_detection.tflite"
landmark_model_path = "/hand_tracking/models/hand_landmark_3d.tflite"
anchors_path = "/hand_tracking/data/anchors.csv"

detector = HandTracker(palm_model_path, landmark_model_path, anchors_path, box_shift=0.2, box_enlarge=1.3)

reader = imageio.get_reader('/video.mp4')
counter = 0

for i, img in enumerate(reader):
    counter += 1
    if img.shape[0] == 256 and img.shape[1] == 256:
        print("Skipping hand detection")
        hands = [{"bbox": np.array([(0, 256), (256, 256), (256, 0), (0, 0)])}]
        hands = [detector.get_landmarks(img, h) for h in hands]
    else:
        hands = detector(img)

    clear_hands = [hand_from_raw(hand["joints"]) for hand in hands]
    name = str(i).zfill(5)
    json.dump(clear_hands, open("/out/" + name + ".json", "w"))

print("Finished", counter)
