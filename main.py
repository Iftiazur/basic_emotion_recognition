# emotion_ui.py
import tkinter as tk
from start_detection import run_emotion_detection

def emotionButton(name_entry):
    person_name = name_entry.get()
    if not person_name:
        person_name = "Anonymous"
    run_emotion_detection(person_name)

root = tk.Tk()
root.title("Emotion Detection UI")

# Label
label = tk.Label(root, text="EMOTION DETECTION PROGRAM", font=("Helvetica", 16))
label.pack(pady=20)

# Entry for the person's name
name_label = tk.Label(root, text="Enter Your Name:")
name_label.pack()
name_entry = tk.Entry(root)
name_entry.pack()

# Button
button = tk.Button(root, text="Start Emotion Detection", command=lambda: emotionButton(name_entry))
button.pack(pady=20)

root.mainloop()
