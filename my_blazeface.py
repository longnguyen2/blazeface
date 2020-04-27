from blazeface import BlazeFace
import torch
import numpy as np
import cv2

class MyBlazeFace(torch.nn.Module):
  def __init__(self, blaze_face, anchors_path):
    super(MyBlazeFace, self).__init__()

    self.anchors = torch.tensor(np.load(anchors_path), dtype=torch.float32, device='cpu')
    self.blaze_face = blaze_face
    self.num_classes = 1
    self.num_anchors = 896
    self.num_coords = 16
    self.score_clipping_thresh = 100.0
    self.x_scale = 128.0
    self.y_scale = 128.0
    self.h_scale = 128.0
    self.w_scale = 128.0
    self.min_score_thresh = 0.75
    self.min_suppression_threshold = 0.3

  def forward(self, x):
    out = self.blaze_face(x)
    detection_boxes = torch.zeros_like(out[0])

    x_center = out[0][..., 0] / self.x_scale * self.anchors[:, 2] + self.anchors[:, 0]
    y_center = out[0][..., 1] / self.y_scale * self.anchors[:, 3] + self.anchors[:, 1]

    w = out[0][..., 2] / self.w_scale * self.anchors[:, 2]
    h = out[0][..., 3] / self.h_scale * self.anchors[:, 3]

    detection_boxes[..., 0] = y_center - h / 2.  # ymin
    detection_boxes[..., 1] = x_center - w / 2.  # xmin
    detection_boxes[..., 2] = y_center + h / 2.  # ymax
    detection_boxes[..., 3] = x_center + w / 2.  # xmax

    for k in range(6):
      offset = 4 + k*2
      keypoint_x = out[0][..., offset    ] / self.x_scale * self.anchors[:, 2] + self.anchors[:, 0]
      keypoint_y = out[0][..., offset + 1] / self.y_scale * self.anchors[:, 3] + self.anchors[:, 1]
      detection_boxes[..., offset    ] = keypoint_x
      detection_boxes[..., offset + 1] = keypoint_y
    
    thresh = self.score_clipping_thresh
    out[1] = out[1].clamp(-thresh, thresh)
    detection_scores = out[1].sigmoid().squeeze(dim=-1)

    mask = detection_scores >= self.min_score_thresh

    output_detections = []
    for i in range(out[1].shape[0]):
      boxes = detection_boxes[i, mask[i]]
      scores = detection_scores[i, mask[i]].unsqueeze(dim=-1)
      output_detections.append(torch.cat((boxes, scores), dim=-1))
    
    filtered_detections = []
    for i in range(len(output_detections)):
      if len(output_detections[i]) == 0:
        faces = []
      else:
        faces = []

        # Sort the detections from highest to lowest score.
        remaining = torch.argsort(output_detections[i][:, 16], descending=True)

        while len(remaining) > 0:
          detection = output_detections[i][remaining[0]]

          # Compute the overlap between the first box and the other 
          # remaining boxes. (Note that the other_boxes also include
          # the first_box.)
          first_box = detection[:4]
          other_boxes = output_detections[i][remaining, :4]
          box_a = first_box.unsqueeze(0)
          box_b = other_boxes
          A = box_a.size(0)
          B = box_b.size(0)
          max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                            box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
          min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                            box_b[:, :2].unsqueeze(0).expand(A, B, 2))
          inter = torch.clamp((max_xy - min_xy), min=0)
          inter = inter[:, :, 0] * inter[:, :, 1]
          area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
          area_b = ((box_b[:, 2]-box_b[:, 0]) *
              (box_b[:, 3]-box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
          union = area_a + area_b - inter
          ious = (inter / union).squeeze(0)

          # If two detections don't overlap enough, they are considered
          # to be from different faces.
          mask = ious > self.min_suppression_threshold
          overlapping = remaining[mask]
          remaining = remaining[~mask]

          # Take an average of the coordinates from the overlapping
          # detections, weighted by their confidence scores.
          weighted_detection = detection.clone()
          if len(overlapping) > 1:
            coordinates = output_detections[i][overlapping, :16]
            scores = output_detections[i][overlapping, 16:17]
            total_score = scores.sum()
            weighted = (coordinates * scores).sum(dim=0) / total_score
            weighted_detection[:16] = weighted
            weighted_detection[16] = total_score / len(overlapping)

          faces.append(weighted_detection)
      faces = torch.stack(faces) if len(faces) > 0 else torch.zeros((0, 17))
      filtered_detections.append(faces)
    
    return filtered_detections[0]

net = BlazeFace().to("cpu")
net.load_state_dict(torch.load("blazeface.pth"))
net.eval()
myNet = MyBlazeFace(net, "anchors.npy").to("cpu")
torch.save(myNet, "myBlazeface.pth")
# from torch.autograd import Variable
# myNet = torch.load("myBlazeface.pth")
# dummy_input = Variable(torch.randn(1, 3, 128, 128)) # nchw
# onnx_filename = "blazeface.onnx"
# torch.onnx.export(myNet, dummy_input,
#                   onnx_filename,
#                   verbose=True)

# import onnx
# from onnx_tf.backend import prepare
# from PIL import Image
# onnx_model = onnx.load(onnx_filename)
# tf_rep = prepare(onnx_model, strict=False)
# print(tf_rep.inputs)
# print(tf_rep.outputs)
# # install onnx-tensorflow from github，and tf_rep = prepare(onnx_model, strict=False)
# # Reference https://github.com/onnx/onnx-tensorflow/issues/167
# #tf_rep = prepare(onnx_model) # whthout strict=False leads to KeyError: 'pyfunc_0'
# # image = Image.open('1face.png')
# # # debug, here using the same input to check onnx and tf.
# # output_pytorch, img_np = modelhandle.process(image)
# # print('output_pytorch = {}'.format(output_pytorch))
# # output_onnx_tf = tf_rep.run(img_np)
# # print('output_onnx_tf = {}'.format(output_onnx_tf))
# # onnx --> tf.graph.pb
# tf_pb_path = onnx_filename + '_graph.pb'
# tf_rep.export_graph(tf_pb_path)

# import tensorflow as tf
# import cv2
# import numpy as np

# with tf.Graph().as_default():
#     graph_def = tf.GraphDef()
#     with open(tf_pb_path, "rb") as f:
#         graph_def.ParseFromString(f.read())
#         tf.import_graph_def(graph_def, name="")
#     with tf.Session() as sess:
#         #init = tf.initialize_all_variables()
#         init = tf.global_variables_initializer()
#         #sess.run(init)
        
#         # print all ops, check input/output tensor name.
#         # uncomment it if you donnot know io tensor names.
#         '''
#         print('-------------ops---------------------')
#         op = sess.graph.get_operations()
#         for m in op:
#             print(m.values())
#         print('-------------ops done.---------------------')
#         '''

#         input_x = sess.graph.get_tensor_by_name("x.1:0") # input
#         outputs1 = sess.graph.get_tensor_by_name('75:0') # 5
#         img = cv2.imread("1face.png")
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         img = np.transpose(img, (2, 0, 1))[np.newaxis,:,:,:].astype(np.float32)
#         img = img / 127.5 - 1.0
#         output_tf_pb = sess.run([outputs1], feed_dict={input_x:img})
#         #output_tf_pb = sess.run([outputs1, outputs2], feed_dict={input_x:np.random.randn(1, 3, 224, 224)})
#         tf.print(output_tf_pb)


# python3 /usr/local/lib/python3.7/site-packages/tensorflow_core/lite/python/tflite_convert.py \
# --output_file my_blazeface.tflite --graph_def_file blazeface.onnx_graph.pb --output_format TFLITE \
# --input_arrays input --input_shapes 1,3,128,128 --output_arrays embeddings \

# python3 /usr/local/lib/python3.7/site-packages/tensorflow_core/lite/python/tflite_convert.py \
# --output_file my_blazeface.tflite --graph_def_file blazeface.onnx_graph.pb --output_format TFLITE \
# --input_arrays x.1 --input_shapes 1,3,128,128 --output_arrays 75

# import tensorflow as tf
# import cv2
# import sys
# import numpy as np

# # # Load TFLite model and allocate tensors.
# interpreter = tf.lite.Interpreter(model_path="my_blazeface.tflite")
# interpreter.allocate_tensors()

# # # Get input and output tensors.
# input_details = interpreter.get_input_details()
# output_details = interpreter.get_output_details()
# print(output_details)
 
# input_data = interpreter.get_tensor(input_details[0]['index'])
# output_data = interpreter.get_tensor(output_details[0]['index'])
# print(input_data.shape)
# print(output_data.shape)