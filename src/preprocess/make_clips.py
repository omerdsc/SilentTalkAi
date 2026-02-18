# src/preprocess/make_clips.py
import os, csv, glob, random
from dataclasses import dataclass
from typing import List, Tuple, Optional

import cv2
import numpy as np
from tqdm import tqdm
import imageio.v2 as imageio
import mediapipe as mp


@dataclass
class CFG:
    project_root: str = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    raw_dir: str = os.path.join(project_root, "data_raw")
    out_dir: str = os.path.join(project_root, "data_processed")
    clips_dir: str = os.path.join(out_dir, "clips")
    manifest: str = os.path.join(out_dir, "manifest.csv")

    labels = ("agri", "evet", "hayir", "su")
    exts = (".mov", ".mp4", ".avi", ".mkv")

    target_frames = 16
    face_size = 112
    padding = 0.20


def read_video_frames(path):
    frames = []
    cap = cv2.VideoCapture(path)
    while True:
        ret, f = cap.read()
        if not ret: break
        frames.append(f)
    cap.release()

    if len(frames) < 5:
        reader = imageio.get_reader(path)
        frames = [cv2.cvtColor(fr, cv2.COLOR_RGB2BGR) for fr in reader]
    return frames


def sample_indices(n, k):
    return np.linspace(0, n-1, k).astype(int)


class FaceCropper:
    def __init__(self):
        self.det = mp.solutions.face_detection.FaceDetection(0,0.5)

    def crop(self, frame, size, pad):
        h,w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.det.process(rgb)
        if not res.detections:
            return None

        d = res.detections[0]
        bb = d.location_data.relative_bounding_box
        x1,y1 = int(bb.xmin*w), int(bb.ymin*h)
        x2,y2 = int((bb.xmin+bb.width)*w), int((bb.ymin+bb.height)*h)

        bw, bh = x2-x1, y2-y1
        x1=max(0,x1-int(bw*pad)); y1=max(0,y1-int(bh*pad))
        x2=min(w-1,x2+int(bw*pad)); y2=min(h-1,y2+int(bh*pad))

        face=frame[y1:y2,x1:x2]
        if face.size==0: return None
        face=cv2.resize(face,(size,size))
        return face.astype(np.float32)/255.


def main():
    cfg=CFG()
    os.makedirs(cfg.clips_dir,exist_ok=True)
    cropper=FaceCropper()
    rows=[]

    for lab in cfg.labels:
        folder=os.path.join(cfg.raw_dir,lab)
        for vp in glob.glob(folder+"/*"):
            if os.path.splitext(vp)[1].lower() not in cfg.exts: continue

            frames=read_video_frames(vp)
            idxs=sample_indices(len(frames),cfg.target_frames)

            clip=[]
            for i in idxs:
                c=cropper.crop(frames[i],cfg.face_size,cfg.padding)
                if c is None: continue
                clip.append(c)

            if len(clip)!=cfg.target_frames: continue

            x=np.stack(clip)
            name=f"{lab}_{os.path.basename(vp)}.npz"
            out=os.path.join(cfg.clips_dir,name)
            np.savez_compressed(out,x=x,label=lab)

            rows.append([out,lab])

    with open(cfg.manifest,"w",newline="") as f:
        csv.writer(f).writerows([["clip","label"]]+rows)

    print("DONE clips:",len(rows))


if __name__=="__main__":
    main()
