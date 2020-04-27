import tensorflow as tf
import cv2
import sys
import numpy as np

interpreter = tf.lite.Interpreter(model_path="face_detection_front.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test model on random input data.
input_shape = input_details[0]['shape']
# input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
# interpreter.set_tensor(input_details[0]['index'], input_data)

interpreter.invoke()

# The function `get_tensor()` returns a copy of the tensor data.
# Use `tensor()` in order to get a pointer to the tensor.
input_data = interpreter.get_tensor(input_details[0]['index'])
output_data = interpreter.get_tensor(output_details[0]['index'])
print(input_data.shape)
print(output_data.shape)

img = cv2.imread("1face.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
img = img / 127.5 - 1.0

interpreter.set_tensor(input_details[0]['index'], img[np.newaxis,:,:,:])
raw_boxes = interpreter.get_tensor(output_details[0]['index'])

anchors = np.load('anchors.npy').astype(np.float32)

boxes = np.zeros_like(raw_boxes)
x_scale = 128.0
y_scale = 128.0
h_scale = 128.0
w_scale = 128.0
min_score_thresh = 0.75
x_center = raw_boxes[..., 0] / x_scale * anchors[:, 2] + anchors[:, 0]
y_center = raw_boxes[..., 1] / y_scale * anchors[:, 3] + anchors[:, 1]

w = raw_boxes[..., 2] / w_scale * anchors[:, 2]
h = raw_boxes[..., 3] / h_scale * anchors[:, 3]

boxes[..., 0] = y_center - h / 2.  # ymin
boxes[..., 1] = x_center - w / 2.  # xmin
boxes[..., 2] = y_center + h / 2.  # ymax
boxes[..., 3] = x_center + w / 2.  # xmax
print(boxes[..., 0].shape)

for k in range(6):
    offset = 4 + k*2
    keypoint_x = raw_boxes[..., offset    ] / x_scale * anchors[:, 2] + anchors[:, 0]
    keypoint_y = raw_boxes[..., offset + 1] / y_scale * anchors[:, 3] + anchors[:, 1]
    boxes[..., offset    ] = keypoint_x
    boxes[..., offset + 1] = keypoint_y

detection_boxes = boxes
mask = detection_boxes >= min_score_thresh
output_boxes = []
for i in range(raw_boxes.shape[0]):
    output_boxes.append(detection_boxes[i, mask[i]])
# print(np.sort(output_boxes[0], axis = 0)[:16])
