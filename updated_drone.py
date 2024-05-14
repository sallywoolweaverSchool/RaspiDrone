import argparse
import logging
import sys
import time
import cv2
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from codrone_edu.drone import *
print("TensorFlow version:", tf.__version__)
print("TensorFlow Hub version:", hub.__version__)
def load_model():
    model_url = "https://tfhub.dev/google/movenet/singlepose/lightning/4"
    model = hub.load(model_url)
    print("Model loaded successfully.")
    return model

def preprocess_image(image, input_size):
    img = tf.image.resize_with_pad(image, input_size[0], input_size[1])
    img = tf.cast(img, dtype=tf.int32)
    img = tf.expand_dims(img, axis=0)
    return img

def draw_keypoints(image, keypoints, confidence_threshold=0.5):
    y, x, c = image.shape
    for keypoint in keypoints:
        ky, kx, kp_conf = keypoint
        if kp_conf > confidence_threshold:
            cv2.circle(image, (int(kx * x), int(ky * y)), 6, (0, 255, 0), -1)
    return image

def run(estimation_model: str, classification_model: str, label_file: str,
        camera_id: int, width: int, height: int, drone: Drone) -> None:
    """Continuously run inference on images acquired from the camera."""
    # Start capturing video input from the camera
    cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    
    model = load_model()
    
    # Variables to calculate FPS
    counter, fps = 0, 0
    start_time = time.time()


    while cap.isOpened():
        success, image = cap.read()
        if not success:
            sys.exit('ERROR: Unable to read from webcam. Please verify your webcam settings.')

        counter += 1
        image = cv2.flip(image, 1)
        input_image = preprocess_image(image, (192, 192))

        # Perform the inference
        outputs = model.signatures['serving_default'](input_image)
        keypoints_with_scores = outputs['output_0'].numpy()

        # Draw keypoints
        keypoints = keypoints_with_scores[0, 0, :, :]
        image = draw_keypoints(image, keypoints)

        left_ankle_y = keypoints[15][0] * image.shape[0]
        right_ankle_y = keypoints[16][0] * image.shape[0]
        nose_y = keypoints[0][0] * image.shape[0]

        # Control the drone based on keypoints
        if left_ankle_y < 100 or right_ankle_y < 100:
            drone.go(0, 0, 0, 50, 0.25)
        elif left_ankle_y > 350 or right_ankle_y > 350:
            drone.go(0, 0, 0, -75, 0.50)
        
        if nose_y < 10:
            drone.land()
        if 100 < nose_y < 200 and (100 < left_ankle_y < 350 and 100 < right_ankle_y < 350):
            drone.flip("front")

        # Calculate the FPS
        if counter % 10 == 0:
            end_time = time.time()
            fps = 10 / (end_time - start_time)
            start_time = time.time()

        # Show the FPS
        fps_text = 'FPS = ' + str(int(fps))
        cv2.putText(image, fps_text, (24, 20), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)
        
        # Display the image
        cv2.imshow('MoveNet', image)

        # Stop the program if the ESC key is pressed
        if cv2.waitKey(1) == 27:
            drone.land()
            break

    cap.release()
    cv2.destroyAllWindows()

def main():
    drone = Drone()
    drone.pair()
    drone.takeoff()
    drone.hover(1)
    print("Drone created")

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', help='Name of estimation model.', required=False, default='movenet_lightning')
    parser.add_argument('--classifier', help='Name of classification model.', required=False)
    parser.add_argument('--label_file', help='Label file for classification.', required=False, default='labels.txt')
    parser.add_argument('--cameraId', help='Id of camera.', required=False, default=0)
    parser.add_argument('--frameWidth', help='Width of frame to capture from camera.', required=False, default=640)
    parser.add_argument('--frameHeight', help='Height of frame to capture from camera.', required=False, default=480)
    args = parser.parse_args()

    run(args.model, args.classifier, args.label_file, int(args.cameraId), args.frameWidth, args.frameHeight, drone)

if __name__ == '__main__':
    main()
