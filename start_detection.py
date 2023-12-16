# main.py
import cv2
import os
from datetime import datetime
from detection import detect_emotions
from show_graph import show_emotion_graph
import matplotlib.pyplot as plt
def run_emotion_detection(person_name):
    cap = cv2.VideoCapture(0)

    # Check if the camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open camera.")
        exit()

    # Create a window to display the output
    cv2.namedWindow("Emotion Detection", cv2.WINDOW_NORMAL)

    # Initialize the loop variables
    emotion_counts = {emotion: 0 for emotion in ["Angry", "Disgusted", "Fearful", "Happy", "Neutral", "Sad", "Surprised"]}
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Detect emotions and display the result
        output_frame = detect_emotions(frame, emotion_counts)
        cv2.imshow("Emotion Detection", output_frame)

        # Check for key presses
        key = cv2.waitKey(1)

        # Break the loop if the 'ESC' key is pressed
        if key == 27:
            save_path = save_emotion_graph(person_name, emotion_counts)
            print(f"Emotion graph saved at: {save_path}")

            break
        # Show bar graph if 'b' key is pressed
        elif key & 0xFF == ord('b'):
            show_emotion_graph(emotion_counts)



    # Release the camera and close all windows
    cap.release()
    cv2.destroyAllWindows()

def save_emotion_graph(person_name, emotion_counts):
    # Create a directory for the person's name if it doesn't exist
    directory = f"emotion_data/{person_name}"

    # Check if the directory already exists
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Generate a unique file name with the current date and time
    date_time_str = datetime.now().strftime("%Y%m%d%H%M%S")
    file_name = f"{date_time_str}_emotion_graph.png"

    # Create a bar chart to display the emotion frequencies
    plt.bar(emotion_counts.keys(), emotion_counts.values())
    plt.xlabel("Emotion")
    plt.ylabel("Frequency")
    plt.title("Emotion Frequency Graph")

    # Save the bar chart in the person's directory
    save_path = os.path.join(directory, file_name)
    plt.savefig(save_path)
    plt.close()  # Close the plot to release resources

    return save_path
