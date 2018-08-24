import numpy as np

def show_summary_data(episodes, rewards, durations):
    print("=" * 30)
    print("Program done")
    print("Ran {} episodes".format(episodes))
    print("Average Reward: {}".format(np.mean(rewards)))
    print("Average Duration: {}".format(np.mean(durations)))

    highlights = {
        "Shortest Episode": np.argmin(durations),
        "Longest Episode": np.argmax(durations),
        "Smallest Reward": np.argmin(rewards),
        "Largest Reward": np.argmax(rewards)
    }

    for description, idx in highlights.items():
        print("=" * 30)
        print(description)
        print("Duration: {} Reward: {}".format(durations[idx], rewards[idx]))