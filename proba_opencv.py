import cv2
import numpy as np

def detect_objects_with_onnx(image_path, model_path, class_labels, confidence_threshold=0.5):
    # Load the ONNX model
    net = cv2.dnn.readNet(model_path)

    # Load the image
    image = cv2.imread(image_path)
    height, width = image.shape[:2]
    print(height, width)
    # Prepare the image for object detection
    blob = cv2.dnn.blobFromImage(image, 1/255 , (640, 640), swapRB=True, mean=(0,0,0), crop= False)
    net.setInput(blob)
    outputs= net.forward()

    print(outputs)
# Path to the image you want to detect objects in
image_path = "380.jpg"

# Path to the ONNX model file
model_path = "best.onnx"

# List of class labels
class_labels = ["Background", "crack"]  # Update with your actual class labels

# Call the function to perform object detection
detect_objects_with_onnx(image_path, model_path, class_labels)
