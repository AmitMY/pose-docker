import sys
sys.path.append("/hand_tracking")

from hand_tracker import HandTracker
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon


palm_model_path = "./models/palm_detection.tflite"
landmark_model_path = "./models/hand_landmark.tflite"
anchors_path = "./data/anchors.csv"

img = cv2.imread('./data/test_img.jpg')[:,:,::-1]

# box_shift determines
detector = HandTracker(palm_model_path, landmark_model_path, anchors_path,
                       box_shift=0.2, box_enlarge=1.3)

kp, box = detector(img)

f,ax = plt.subplots(1,1, figsize=(10, 10))
ax.scatter(kp[:,0], kp[:,1])
ax.add_patch(Polygon(box, color="#00ff00", fill=False))

f.savefig("/api/test.png")