import cv2


def GetVideo(source):
    capture = cv2.VideoCapture(source)
    if isinstance(source, str) and source.startswith("rtsp"):
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    return capture
