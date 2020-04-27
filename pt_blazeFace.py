import numpy as np
import torch
import cv2
from blazeface import BlazeFace

gpu = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = BlazeFace().to(gpu)
net.load_weights("blazeface.pth")
net.load_anchors("anchors.npy")

net.min_score_thresh = 0.75
net.min_suppression_threshold = 0.3

img = cv2.imread("1face.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

detections = net.predict_on_image(img)