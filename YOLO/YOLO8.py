import cv2 # type: ignore
from ultralytics import YOLO # type: ignore
import matplotlib.pyplot as plt # type: ignore

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')

# Image path
image_path = 'Screenshot (37).png'

# Perform object detection
results = model(image_path)

# Print results
for result in results:
    print(result)

# Get the first result image with bounding boxes
result_img = results[0].plot()

# Display the result image using OpenCV
cv2.imshow("Detected Image", result_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Alternatively, display the result image using Matplotlib
result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
plt.imshow(result_img_rgb)
plt.axis('off')  # Hide axes
plt.show()