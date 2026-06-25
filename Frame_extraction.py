import cv2

def extract_frames(video_path,
                   max_frames=10):

    cap = cv2.VideoCapture(video_path)

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    step = max(1, total_frames // max_frames)

    frames = []
    idx = 0

    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:
            break

        if idx % step == 0:
            frames.append(frame)

        idx += 1

        if len(frames) >= max_frames:
            break

    cap.release()

    return frames