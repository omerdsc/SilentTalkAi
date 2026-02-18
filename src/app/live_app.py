# src/app/live_app.py
import os,cv2,torch
import numpy as np
from collections import deque
import streamlit as st
from streamlit_webrtc import webrtc_streamer,VideoProcessorBase,WebRtcMode
import mediapipe as mp
from torchvision.models.video import r2plus1d_18

ROOT=os.path.abspath(os.path.join(os.path.dirname(__file__),"..",".."))
CKPT=os.path.join(ROOT,"models","best.pt")

device="cuda" if torch.cuda.is_available() else "cpu"

ckpt=torch.load(CKPT,map_location=device)
labels=ckpt["labels"]

model=r2plus1d_18(weights=None)
model.fc=torch.nn.Linear(model.fc.in_features,len(labels))
model.load_state_dict(ckpt["model"])
model=model.to(device).eval()

det=mp.solutions.face_detection.FaceDetection(0,0.5)

def crop(frame):
    h,w=frame.shape[:2]
    res=det.process(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
    if not res.detections: return None
    bb=res.detections[0].location_data.relative_bounding_box
    x1,y1=int(bb.xmin*w),int(bb.ymin*h)
    x2,y2=int((bb.xmin+bb.width)*w),int((bb.ymin+bb.height)*h)
    face=frame[y1:y2,x1:x2]
    if face.size==0:return None
    return cv2.resize(face,(112,112))/255.

buf=deque(maxlen=16)

class Cam(VideoProcessorBase):
    def recv(self,frame):
        img=frame.to_ndarray(format="bgr24")
        f=crop(img)
        if f is not None: buf.append(f)

        if len(buf)==16:
            x=torch.tensor(np.stack(buf)).permute(3,0,1,2).unsqueeze(0).float().to(device)
            with torch.no_grad():
                p=torch.softmax(model(x),1)[0]
            lbl=labels[p.argmax()]
            cv2.putText(img,lbl,(10,40),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

        return frame.from_ndarray(img,format="bgr24")

st.title("SilentTalk 3D CNN Live")
webrtc_streamer(key="cam",video_processor_factory=Cam,mode=WebRtcMode.SENDRECV)
