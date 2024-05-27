import cv2
import cvlib as cv
from cvlib.object_detection import draw_bbox
import numpy as np

# Open webcam
video = cv2.VideoCapture(0)

# Check if the webcam opened successfully
if not video.isOpened():
    print("Error: Unable to open webcam.")
else:
    print("Press 'q' to quit.")

# Loop to capture frames from webcam
while video.isOpened():
    # Read a frame from the webcam
    ret, frame = video.read()
    
    # If frame reading was successful
    if ret:
        # Perform object detection on the frame
        objects, labels, confidences = cv.detect_common_objects(frame, confidence=0.5, model='yolov3', enable_gpu=False)
        
        # Draw bounding boxes and labels on the frame
        output_frame = draw_bbox(frame, objects, labels, confidences)
        
        # Calculate centroids of detected objects
        centroids = []
        for obj in objects:
            x1, y1, x2, y2 = obj
            centroid_x = (x1 + x2) // 2
            centroid_y = (y1 + y2) // 2
            centroids.append((centroid_x, centroid_y))
        
        # Calculate distances between all pairs of centroids (in pixels)
        for i in range(len(centroids)):
            for j in range(i+1, len(centroids)):
                centroid1, centroid2 = centroids[i], centroids[j]
                distance_pixels = np.linalg.norm(np.array(centroid1) - np.array(centroid2))
                cv2.putText(output_frame, f"{distance_pixels:.2f} pixels", (int((centroid1[0]+centroid2[0])/2), int((centroid1[1]+centroid2[1])/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Display the frame with bounding boxes, labels, and distances
        cv2.imshow('Object Detection', output_frame)
        
        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Release the webcam and close windows
video.release()
cv2.destroyAllWindows()
