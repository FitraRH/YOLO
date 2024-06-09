import numpy as np
import cv2

# YOLO model files
yolo_weight = "yolov4.weights"
yolo_config = "yolov4.cfg"
coco_labels = "coco.names"
net = cv2.dnn.readNet(yolo_weight, yolo_config)

# Load colo object names file
classes = []
with open(coco_labels, "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Capture video from webcam
cap = cv2.VideoCapture(0)

while True:
    ret, img = cap.read()
    height, width, _ = img.shape

    # Convert image to blob
    blob = cv2.dnn.blobFromImage(img, 1 / 255, (width, height), (0, 0, 0), True, crop=False)

    # Set input for YOLO object detection
    net.setInput(blob)

    # Find names of all layers
    layer_names = net.getLayerNames()

    # Find names of three output layers
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    # Send blob data to forward pass
    outs = net.forward(output_layers)

    # Extract information on the screen
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            # Extract score value
            scores = detection[5:]

            # Object ID
            class_id = np.argmax(scores)

            # Confidence score for each object ID
            confidence = scores[class_id]

            # If confidence > 0.5 and class_id == 0
            if confidence > 0.5:
                # Extract values to draw bounding box
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Draw bounding box with text for each object
    font = cv2.FONT_HERSHEY_DUPLEX
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence_label = int(confidences[i] * 100)
            color = colors[i]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, f'{label} {confidence_label}', (x - 25, y + 65), font, 1, color, 2)

    cv2.imshow("Webcam", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()