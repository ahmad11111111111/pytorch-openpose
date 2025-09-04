import cv2
import copy
import numpy as np
import torch
import sys
import time

from src import model
from src import util
from src.body import Body
from src.hand import Hand

body_estimation = Body('model/body_pose_model.pth')
hand_estimation = Hand('model/hand_pose_model.pth')

# Report selected backend without invoking CUDA-only APIs
if torch.cuda.is_available():
    device_str = 'cuda'
elif getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
    device_str = 'mps'
else:
    device_str = 'cpu'
print(f"Torch device: {device_str}")

def open_camera():
    # Prefer AVFoundation on macOS for better permissions/capture behavior
    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    return cap

cap = open_camera()
if not cap.isOpened():
    print("OpenCV: could not open camera. On macOS, grant Camera permission to your terminal/VS Code in System Settings > Privacy & Security > Camera.")
    sys.exit(1)

# Lower resolution for speed (e.g., 320x240) and request high FPS
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
cap.set(cv2.CAP_PROP_FPS, 30)
while True:
    ret, oriImg = cap.read()
    if not ret or oriImg is None:
        time.sleep(0.01)
        continue
    candidate, subset = body_estimation(oriImg)
    canvas = copy.deepcopy(oriImg)
    canvas = util.draw_bodypose(canvas, candidate, subset)

    # detect hand
    hands_list = util.handDetect(candidate, subset, oriImg)

    # Fast OpenCV hand drawing
    for x, y, w, is_left in hands_list:
        peaks = hand_estimation(oriImg[y:y+w, x:x+w, :])
        peaks[:, 0] = np.where(peaks[:, 0]==0, peaks[:, 0], peaks[:, 0]+x)
        peaks[:, 1] = np.where(peaks[:, 1]==0, peaks[:, 1], peaks[:, 1]+y)
        # Draw hand pose directly on canvas
        canvas = util.draw_handpose_by_opencv(canvas, peaks)

    cv2.imshow('demo', canvas)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

