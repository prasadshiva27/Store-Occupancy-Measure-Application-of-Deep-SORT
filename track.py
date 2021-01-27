from absl import flags
import sys
FLAGS = flags.FLAGS
FLAGS(sys.argv)

import time
import numpy as np
import cv2
import matplotlib.pyplot as plt

import tensorflow as tf
from yolov3_tf2.models import YoloV3
from yolov3_tf2.dataset import transform_images
from yolov3_tf2.utils import convert_boxes

from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet

class_names = [c.strip() for c in open('./data/labels/coco.names').readlines()]
yolo = YoloV3(classes=len(class_names))
yolo.load_weights('./weights/yolov3.tf')

max_cosine_distance = 0.5
nn_budget = None
nms_max_overlap = 0.8


model_filename = 'model_data/mars-small128.pb'
encoder = gdet.create_box_encoder(model_filename, batch_size=1)
metric = nn_matching.NearestNeighborDistanceMetric('cosine', max_cosine_distance, nn_budget)
tracker = Tracker(metric)

vid = cv2.VideoCapture('./data/video/mall_leave.mpg')

codec = cv2.VideoWriter_fourcc(*'XVID')
vid_fps =int(vid.get(cv2.CAP_PROP_FPS))
vid_width,vid_height = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter('./data/video/results.avi', codec, vid_fps, (vid_width, vid_height))

from _collections import deque
pts = [deque(maxlen=30) for _ in range(1000)]

counter = []
W = None
H = None

peopleOut = 128
peopleIn = 226
inStoreCount = peopleIn - peopleOut


while True:
    _, img = vid.read()
    if img is None:
        print('Completed')
        break
    if W is None or H is None:
        (H, W) = img.shape[:2]

    img_in = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_in = tf.expand_dims(img_in, 0)
    img_in = transform_images(img_in, 416)

    t1 = time.time()

    if inStoreCount <100:
        cv2.circle(img, (W//2 + 15,25), 8, (0, 255, 0), -1)
    else:
        cv2.circle(img, (W//2 + 15,25), 8, (0, 0, 255), -1)
        print("Store is Full")
    
    # boxes,    3D shape (1,100,4) --> maximum of 100 bounding boxes in image with X,Y,WIDTH,HEIGHT as values
    # scores,   2D (1,100)        --> Confidence scores of detected object
    # classes,  2D Shape (1,100) --> Detected classes 
    # nums,     1D shape            --> Total no of detected objects

    boxes, scores, classes, nums = yolo.predict(img_in)

    classes = classes[0]
    names = []
    for i in range(len(classes)):
        names.append(class_names[int(classes[i])])
    names = np.array(names)
    converted_boxes = convert_boxes(img, boxes[0])
    features = encoder(img, converted_boxes)

    detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in
                  zip(converted_boxes, scores[0], names, features)]

    boxs = np.array([d.tlwh for d in detections])
    scores = np.array([d.confidence for d in detections])
    classes = np.array([d.class_name for d in detections])
    indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
    detections = [detections[i] for i in indices]

    tracker.predict()
    tracker.update(detections,H)
    

    cmap = plt.get_cmap('tab20b')
    colors = [cmap(i)[:3] for i in np.linspace(0,1,20)]

    current_count = int(0)

    for track in tracker.tracks:

        info = [
            ("People Entered", peopleIn),
            ("People Exited", peopleOut)
                ]
        for (i, (k, v)) in enumerate(info):
            text = "{}: {}".format(k, v)
            # cv2.putText(img, text, (10, H - ((i * 20) + 20)),cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
            cv2.putText(img, text, (int(W*0.70), 50 - ((i * 10) + 20)),0, 0.3, (255, 0, 0), 1)
        # cv2.putText(img, "Estimated People inside Store:" + str(inStoreCount), (300,30), 0, 1, (0,0,255), 2)
        cv2.putText(img, "Total Occupancy:" + str(inStoreCount), (int(W*0.70),50), 0, 0.3, (255,0,0), 1)


        if not track.is_confirmed() or track.time_since_update >1:
            continue
        bbox = track.to_tlbr()
        class_name= track.get_class()
        color = colors[int(track.track_id) % len(colors)]
        color = [i * 255 for i in color]
        
        c1 = (int(bbox[0]) + int(bbox[2]))/2
        c2 = (int(bbox[1]) + int(bbox[3]))/2
        centerPoint = (int(c1), int(c2))
        cv2.putText(img, str(track.track_id),centerPoint,0, 5e-3 * 200, (0,0,255),1)
        cv2.circle(img, centerPoint, 4, (0, 255, 255), -1)


        center = (int(((bbox[0]) + (bbox[2]))/2), int(((bbox[1])+(bbox[3]))/2))
        
        center_y = int(((bbox[1])+(bbox[3]))/2)

        # print(track.track_id, track.stateOutMetro)
        # if class_name  == 'person':
        # print(class_name)
                
        if track.stateOutMetro == 1 and (int(H * 0.45) - (int(bbox[3]) + int(bbox[1]))/2 > 0) and track.noConsider == False and class_name  == 'person':
            peopleIn += 1
            track.stateOutMetro = 0
            track.noConsider = True
            # cv2.line(img, (0, H // 2 +50), (W, H // 2 +50), (0, 0, 0), 2)
            cv2.line(img, (W // 2, int(H * 0.45)), (W, int(H * 0.45)), (0, 0, 0), 1)
            print("Track_ID:Entered",track.track_id)
            inStoreCount = peopleIn - peopleOut
            if inStoreCount <100:
                cv2.circle(img, (W//2 + 15,25), 8, (0, 255, 0), -1)
            else:
                cv2.circle(img, (W//2 + 15,25), 8, (0, 0, 255), -1)
                print("Store is Full")
            

        if track.stateOutMetro == 0 and (int(H * 0.45) - (int(bbox[3]) + int(bbox[1]))/2 <= 0) and track.noConsider == False and class_name  == 'person':
            peopleOut += 1
            track.stateOutMetro = 1
            track.noConsider = True
            # cv2.line(img, (0, H // 2 +50), (W, H // 2 +50), (0, 0, 0), 2)
            cv2.line(img, (W // 2, int(H * 0.45)), (W, int(H * 0.45)), (0, 0, 0), 1)
            print("Track_ID:Exited",track.track_id)
            inStoreCount = peopleIn - peopleOut
            if inStoreCount <100:
                cv2.circle(img, (W//2 + 15,25), 8, (0, 255, 0), -1)
            else:
                cv2.circle(img, (W//2 + 15,25), 8, (0, 0, 255), -1)
                print("Store is Full")

        # cv2.line(img, (0, H // 2 +50), (W, H // 2 + 50), (0, 0, 255), 2)
        cv2.line(img, (W // 2, int(H * 0.45)), (W, int(H * 0.45)), (0, 0, 255), 1)
        info = [
            ("People Entered", peopleIn),
            ("People Exited", peopleOut)
                ]
        for (i, (k, v)) in enumerate(info):
            text = "{}: {}".format(k, v)
            cv2.putText(img, text, (int(W*0.70), 50 - ((i * 10) + 20)),0, 0.3, (255, 0, 0), 1)

        
        pts[track.track_id].append(center)

        for j in range(1, len(pts[track.track_id])):
            if pts[track.track_id][j-1] is None or pts[track.track_id][j] is None:
                continue
            thickness = int(np.sqrt(64/float(j+1))*2)
            # cv2.line(img, (pts[track.track_id][j-1]), (pts[track.track_id][j]), color, thickness)

    fps = 1./(time.time()-t1)
    # cv2.putText(img, "FPS: {:.2f}".format(fps), (0,30), 0, 1, (0,0,255), 2)
    cv2.putText(img, "Total Occupancy:" + str(inStoreCount), (int(W*0.70),50), 0, 0.3, (255,0,0), 1)
    # cv2.resizeWindow('output', 1024, 768)
    cv2.resizeWindow('output', 500, 300)
    cv2.imshow('output', img)
    out.write(img)

    if cv2.waitKey(1) == ord('q'):
        break
vid.release()
out.release()
cv2.destroyAllWindows()
