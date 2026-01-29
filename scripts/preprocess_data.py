import os
import glob
import cv2
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import argparse

def preprocess_video(video_path, save_path, resolution=(128, 128)):
    cap = cv2.VideoCapture(video_path)
    frames_list = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret: break
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, c = frame.shape
            min_dim = min(h, w)
            start_h = (h - min_dim) // 2
            start_w = (w - min_dim) // 2
            frame = frame[start_h:start_h+min_dim, start_w:start_w+min_dim]
            
            frame_pil = Image.fromarray(frame).resize(resolution, Image.BILINEAR)
            frame_arr = np.array(frame_pil)
            frames_list.append(frame_arr)
            
    finally:
        cap.release()
        
    if len(frames_list) > 0:
        frames_np = np.stack(frames_list)
        frames_tensor = torch.from_numpy(frames_np).permute(0, 3, 1, 2)
        torch.save(frames_tensor, save_path)
        return True
    return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/raw", help="Source directory")
    parser.add_argument("--out_dir", type=str, default="data/processed", help="Destination directory")
    parser.add_argument("--size", type=int, default=128, help="Target resolution")
    parser.add_argument("--workers", type=int, default=8, help="Number of worker processes")
    args = parser.parse_args()
    
    splits = ["train", "val", "test"]
    
    for split in splits:
        src_dir = os.path.join(args.data_dir, split)
        dst_dir = os.path.join(args.out_dir, split)
        
        if not os.path.exists(src_dir):
            continue
            
        os.makedirs(dst_dir, exist_ok=True)
        
        files = sorted(glob.glob(os.path.join(src_dir, "*.mp4")))
        print(f"Processing {split}: {len(files)} files...")
        
        from multiprocessing import Pool
        from functools import partial
        
        resolution = (args.size, args.size)
        
        tasks = []
        for f in files:
            fname = os.path.basename(f).replace(".mp4", ".pt")
            save_path = os.path.join(dst_dir, fname)
            if not os.path.exists(save_path):
                tasks.append((f, save_path))
        
        if len(tasks) == 0:
            print(f"All files in {split} already processed.")
            continue

        func = partial(process_wrapper, resolution=resolution)

        with Pool(args.workers) as p:
            for _ in tqdm(p.imap_unordered(func, tasks), total=len(tasks)):
                pass

def process_wrapper(task, resolution):
    return preprocess_video(task[0], task[1], resolution)


if __name__ == "__main__":
    main()
