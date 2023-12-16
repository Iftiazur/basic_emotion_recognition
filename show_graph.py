import matplotlib.pyplot as plt

def show_emotion_graph(emotion_counts):
    # Create a bar chart to display the emotion frequencies
    plt.bar(emotion_counts.keys(), emotion_counts.values())
    plt.xlabel("Emotion")
    plt.ylabel("Frequency")
    plt.title("Emotion Frequency Graph")
    plt.show()
