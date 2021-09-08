#importa biblioteca de streeming#

from flask_opencv_streamer.streamer import Streamer

#importa biblioteca de deteccao objetos em movimento#
from cvlib.object_detection import draw_bbox


import cv2
import cvlib as cv

# http://insecam.org/en/view/912189/#

port = 9092
require_login = False
streamer = Streamer(port, require_login)

endpoint_cam = "http://189.131.16.109:84/mjpg/video.mjpg" # ok 

def main():
    video_capture = cv2.VideoCapture(endpoint_cam)

    while True:
        _, frame = video_capture.read()
        bbox, label, conf = cv.detect_common_objects(frame, confidence=0.50, model='yolov4')
        print(bbox,label)
        for tag in label:
            if tag == "person":
                frame = draw_bbox(frame, bbox, label, conf)
        
        streamer.update_frame(frame)
        if not streamer.is_streaming:
            streamer.start_streaming()

        cv2.waitKey(30)    

if __name__ == "__main__":
    main()
