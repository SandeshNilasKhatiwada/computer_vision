import cv2
import numpy as np
import tensorflow as tf
import platform
import argparse

def load_model(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

def load_labels(label_path):
    with open(label_path, "r") as f:
        return [line.strip() for line in f.readlines()]

def preprocess_image(image, input_size):
    input_data = cv2.resize(image, input_size)
    input_data = np.expand_dims(input_data, axis=0)
    input_data = input_data.astype(np.uint8)
    return input_data

def detect_traffic_light(frame, interpreter, input_details, output_details, labels):
    input_data = preprocess_image(frame, (300, 300))
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    boxes = interpreter.get_tensor(output_details[0]['index'])[0]  # Bounding box coordinates
    classes = interpreter.get_tensor(output_details[1]['index'])[0]  # Class index
    scores = interpreter.get_tensor(output_details[2]['index'])[0]  # Confidence scores

    for i in range(len(scores)):
        if scores[i] > 0.5 and int(classes[i]) < len(labels) and labels[int(classes[i])] == 'traffic light':
            ymin, xmin, ymax, xmax = boxes[i]
            xmin = int(xmin * frame.shape[1])
            xmax = int(xmax * frame.shape[1])
            ymin = int(ymin * frame.shape[0])
            ymax = int(ymax * frame.shape[0])
            return (xmin, ymin, xmax, ymax)
    return None

def detect_color(roi):
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    red_lower = np.array([136, 87, 111], np.uint8)
    red_upper = np.array([180, 255, 255], np.uint8)
    green_lower = np.array([25, 52, 72], np.uint8)
    green_upper = np.array([102, 255, 255], np.uint8)
    yellow_lower = np.array([20, 100, 100], np.uint8)
    yellow_upper = np.array([30, 255, 255], np.uint8)

    red_mask = cv2.inRange(hsv, red_lower, red_upper)
    green_mask = cv2.inRange(hsv, green_lower, green_upper)
    yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)

    if cv2.countNonZero(red_mask) > 0:
        return "Red"
    elif cv2.countNonZero(yellow_mask) > 0:
        return "Yellow"
    elif cv2.countNonZero(green_mask) > 0:
        return "Green"
    else:
        return "Unknown"

def main(model_path, label_path):
    interpreter = load_model(model_path)
    labels = load_labels(label_path)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    if platform.system() == 'Darwin':  # macOS
        cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
    else:  # Assuming Linux (Raspberry Pi)
        cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Failed to initialize the camera.")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        bbox = detect_traffic_light(frame, interpreter, input_details, output_details, labels)
        if bbox:
            xmin, ymin, xmax, ymax = bbox
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            roi = frame[ymin:ymax, xmin:xmax]
            light_color = detect_color(roi)
            cv2.putText(frame, light_color, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Traffic Light Detection", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # Exit if 'ESC' is pressed
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='Path to the TFLite model file')
    parser.add_argument('--labels', required=True, help='Path to the label map file')
    args = parser.parse_args()

    main(args.model, args.labels)
