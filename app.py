import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import cv2 
import numpy as np 
from img_process_predict import img_live
import pickle
import av
import threading

with open('SVC_model.pkl', 'rb') as f:
    svm = pickle.load(f)

RTC_CONFIGURATION = RTCConfiguration(
    {
      "RTCIceServer": [{
        "urls": "turn:openrelay.metered.ca:80",
        "username": "openrelayproject",
        "credential": "openrelayproject",
      }]
    }
)

st.title("Signals")

class VideoProcessor(VideoProcessorBase):
        def __init__(self):
            self.model_lock = threading.Lock()

        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            data = img_live(img)
            data = np.array(data)
            y_pred = svm.predict(data.reshape(-1,63))
            font = cv2.FONT_HERSHEY_SIMPLEX
            position = (50, 100)
            fontScale = 3
            color = (0, 0, 0)
            thickness = 5
            letter = str(y_pred[0])
            frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            frame = cv2.putText(frame, letter, position, font, 
                            fontScale, color, thickness, cv2.LINE_AA)

            return av.VideoFrame.from_ndarray(frame)

webrtc_streamer(
        key="example",
        video_processor_factory=VideoProcessor,
        rtc_configuration= RTC_CONFIGURATION ,
        media_stream_constraints={
            "video": True,
            "audio": False
        }
    )

        
        
        
        

