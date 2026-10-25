"""Decode exact native RGB frame intervals without evaluation-global side effects."""


def frames(task):
    import cv2
    cap = cv2.VideoCapture(task['video'])
    assert cap.isOpened()
    assert cap.set(cv2.CAP_PROP_POS_FRAMES, task['start_frame'])
    assert round(cap.get(cv2.CAP_PROP_POS_FRAMES)) == task['start_frame']
    try:
        for position in range(task['video_frames']):
            ok, image = cap.read()
            assert ok
            assert round(cap.get(cv2.CAP_PROP_POS_FRAMES)) == task['start_frame'] + position + 1
            yield position, cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    finally:
        cap.release()
