import cv2
import numpy as np
import argparse
import imutils
import time
import os

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
    help="path to input video")
ap.add_argument("-o", "--output", required=True,
    help="path to output video")
ap.add_argument("-y", "--yolo", required=True,
    help="base path to YOLO directory")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
    help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3,
    help="threshold when applying non-maxima suppression")
args = vars(ap.parse_args())

# load the COCO class labels our YOLO model was trained on
labelsPath = os.path.sep.join([args["yolo"], "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

# derive the paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join([args["yolo"], "yolov4.weights"])
configPath = os.path.sep.join([args["yolo"], "yolov4.cfg"])

# load our YOLO object detector trained on COCO dataset (80 classes)
# and determine only the *output* layer names that we need from YOLO
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln = net.getUnconnectedOutLayersNames()

# initialize the video stream, pointer to output video file, and
# frame dimensions
vs = cv2.VideoCapture(args["input"])
writer = None
(W, H) = (None, None)

# Initialize variables for optical flow
prev_gray = None
prev_frame = None
prev_pts = None
lk_params = dict(winSize=(15, 15), maxLevel=2,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Initialize variables for individual object tracking
objects = {}
object_id = 1

# Function to calculate optical flow and estimate velocity
def estimate_velocity(frame, prev_frame, prev_gray, prev_pts, box):
    if prev_gray is None or prev_pts is None or prev_pts.shape[0] == 0:
        return 0.0

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calculate optical flow using previous tracking points
    next_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pts, None, **lk_params)

    # Convert status to a boolean array
    status = status.astype(np.bool)

    # Get the coordinates of the object's center
    x, y, w, h = box
    center_x = x + w // 2
    center_y = y + h // 2

    # Filter valid tracking points around the object's center
    valid_pts = next_pts[status[:, 0] & ((next_pts[:, 0, 0] >= center_x - w // 2) & (next_pts[:, 0, 0] <= center_x + w // 2) &
                                         (next_pts[:, 0, 1] >= center_y - h // 2) & (next_pts[:, 0, 1] <= center_y + h // 2))]
    valid_prev_pts = prev_pts[status[:, 0] & ((next_pts[:, 0, 0] >= center_x - w // 2) & (next_pts[:, 0, 0] <= center_x + w // 2) &
                                               (next_pts[:, 0, 1] >= center_y - h // 2) & (next_pts[:, 0, 1] <= center_y + h // 2))]

    # Check if there are enough valid points
    if len(valid_pts) < 2 or len(valid_prev_pts) < 2:
        return 0.0

    # Calculate the average velocity
    velocities = np.linalg.norm(valid_pts - valid_prev_pts, axis=1)
    average_velocity = np.mean(velocities)

    return average_velocity


# loop over frames from the video file stream
while True:
    # read the next frame from the file
    (grabbed, frame) = vs.read()

    # if the frame was not grabbed, then we have reached the end
    # of the stream
    if not grabbed:
        break

    # if the frame dimensions are empty, grab them
    if W is None or H is None:
        (H, W) = frame.shape[:2]

    # construct a blob from the input frame and then perform a forward
    # pass of the YOLO object detector, giving us our bounding boxes
    # and associated probabilities
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
                                 swapRB=True, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(ln)

    # initialize our lists of detected bounding boxes, confidences,
    # and class IDs, respectively
    boxes = []
    confidences = []
    classIDs = []

    # loop over each of the layer outputs
    for output in layerOutputs:
        # loop over each of the detections
        for detection in output:
            # extract the class ID and confidence (i.e., probability)
            # of the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            # filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if confidence > args["confidence"]:
                # scale the bounding box coordinates back relative to
                # the size of the image, keeping in mind that YOLO
                # actually returns the center (x, y)-coordinates of
                # the bounding box followed by the boxes' width and
                # height
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                # use the center (x, y)-coordinates to derive the top
                # and and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # update our list of bounding box coordinates,
                # confidences, and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    # apply non-maxima suppression to suppress weak, overlapping
    # bounding boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"],
                            args["threshold"])

    # ensure at least one detection exists
    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            # draw a bounding box rectangle and label on the frame
            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
            cv2.putText(frame, text, (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Track individual objects and estimate velocity
            object_box = (x, y, x + w, y + h)
            if object_id not in objects:
                objects[object_id] = {'box': object_box, 'velocities': [0.0], 'prev_pts': None}
            else:
                prev_pts = objects[object_id]['prev_pts']
                velocity = estimate_velocity(frame, prev_frame, prev_gray, prev_pts, object_box)
                objects[object_id]['velocities'].append(velocity)

            # Display object information with box and velocity
            object_label = f"Object {object_id}"
            object_velocity = np.mean(objects[object_id]['velocities'])
            object_info = f"Velocity: {object_velocity:.2f}"
            cv2.putText(frame, object_label, (x, y - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.putText(frame, object_info, (x + int(w/2) - 50, y + int(h/2)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Update tracking points and object region for the next frame
            objects[object_id]['prev_pts'] = cv2.goodFeaturesToTrack(prev_gray, maxCorners=100, qualityLevel=0.3,
                                                                      minDistance=7, blockSize=7)

            # Increment object ID for the next object
            object_id += 1

    # check if the video writer is None
    if writer is None:
        # initialize our video writer
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args["output"], fourcc, 30,
                                 (frame.shape[1], frame.shape[0]), True)

    # write the output frame to disk
    writer.write(frame)

    # show the output frame
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    # Update previous frame and grayscale frame
    prev_frame = frame.copy()
    prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# release the file pointers
vs.release()
writer.release()
cv2.destroyAllWindows()
