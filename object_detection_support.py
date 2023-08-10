import cv2
from tflite_support.task import core
from tflite_support.task import processor
from tflite_support.task import vision

# Create the base options
base_options = core.BaseOptions(file_name='detect.tflite')

# Create the detection options
detection_options = processor.DetectionOptions(max_results=2, score_threshold=0.3)

# Create the vision.ObjectDetectorOptions
options = vision.ObjectDetectorOptions(base_options=base_options, detection_options=detection_options)

# Create the ObjectDetector
detector = vision.ObjectDetector.create_from_options(options)
