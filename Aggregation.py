import numpy as np

def video_prediction(
    frame_probs
):

    return np.mean(
        frame_probs
    )