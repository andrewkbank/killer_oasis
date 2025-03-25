"""
This file takes a gameplay mp4 and a keyboard/mouse-recording file
and compiles it into a gameplay overlay video
This file should mostly be used for debugging
"""
import cv2
import torch
import numpy as np

ACTION_KEYS = [
    "inventory", "ESC", "hotbar.1", "hotbar.2", "hotbar.3", "hotbar.4", "hotbar.5", "hotbar.6", "hotbar.7", "hotbar.8", "hotbar.9", 
    "forward", "back", "left", "right", "jump", "sneak", "sprint", "swapHands", "attack", "use", "pickItem", "drop", "cameraX", "cameraY"
]

def load_actions(action_path):
    return torch.load(action_path, weights_only=False)

def draw_overlay(frame, actions, frame_idx, width, height):
    overlay = frame.copy()
    
    for i, key in enumerate(ACTION_KEYS):
        if key in ["cameraX", "cameraY"]:
            continue  # Camera movement will be visualized differently
        
        value = actions[frame_idx].get(key, 0)
        color = (0, 255, 0) if value else (255, 0, 0)
        cv2.putText(overlay, f"{key}: {value}", (10, 30 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    return overlay

def overlay_video(video_path, action_path, output_path):
    actions = load_actions(action_path)
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error opening video file.")
        return
    
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    frame_idx = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        overlay_frame = draw_overlay(frame, actions, frame_idx, width, height)
        out.write(overlay_frame)
        frame_idx += 1
    
    cap.release()
    out.release()
    print(f"Overlay video saved to {output_path}")

# Example usage:
overlay_video("./recording_data/Player729-f153ac423f61-20210806-224813.chunk_000.mp4", "./recording_data/Player729-f153ac423f61-20210806-224813.chunk_000.actions.pt", "overlay_output/replay_with_overlay.mp4")
