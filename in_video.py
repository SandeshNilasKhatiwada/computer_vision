import cv2
import cvlib as cv
from cvlib.object_detection import draw_bbox
import numpy as np

# Open video file
video_file = 'video.webm'
video = cv2.VideoCapture(video_file)

# Check if the video file opened successfully
if not video.isOpened():
    print("Error: Unable to open video file.")
    exit()

print("Press 'q' to quit.")

# Use a lighter model for object detection
model = 'yolov3-tiny'

# Metrics variables
frame_count = 0
# Loop to capture frames from video
while video.isOpened():
    # Read a frame from the video
    ret, frame = video.read()
    frame_count += 1
    
    if not ret:
        break

    # Perform object detection on the frame

    objects, labels, confidences = cv.detect_common_objects(frame, confidence=0.5, model=model, enable_gpu=False)
    
    # Draw bounding boxes and labels on the frame
    output_frame = draw_bbox(frame, objects, labels, confidences)
    
    # Calculate centroids of detected objects
    centroids = [( (x1 + x2) // 2, (y1 + y2) // 2 ) for x1, y1, x2, y2 in objects]
    
    # Calculate distances between all pairs of centroids (in pixels)
    for i, centroid1 in enumerate(centroids):
        for centroid2 in centroids[i+1:]:
            distance_pixels = np.linalg.norm(np.array(centroid1) - np.array(centroid2))
            mid_point = (int((centroid1[0] + centroid2[0]) / 2), int((centroid1[1] + centroid2[1]) / 2))
            cv2.putText(output_frame, f"{distance_pixels:.2f} pixels", mid_point, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Display the frame with bounding boxes, labels, and distances
    cv2.imshow('Object Detection', output_frame)
    

 
    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video and close windows
video.release()
cv2.destroyAllWindows()
