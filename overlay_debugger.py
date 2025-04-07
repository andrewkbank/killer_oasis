"""
This file takes a gameplay mp4 and a keyboard/mouse-recording file
and compiles it into a gameplay overlay video
This file should mostly be used for debugging
"""
import cv2
import torch
import numpy as np
# {'ESC': 0, 'back': 0, 'drop': 0, 'forward': 0, 'hotbar.1': 0, 'hotbar.2': 0, 'hotbar.3': 0, 'hotbar.4': 0, 'hotbar.5': 0, 'hotbar.6': 0, 'hotbar.7': 0, 'hotbar.8': 0, 'hotbar.9': 0, 
# 'inventory': 0, 'jump': 0, 'left': 0, 'right': 0, 'sneak': 0, 'sprint': 0, 'swapHands': 0, 'camera': array([40, 40]), 'attack': 0, 'use': 0, 'pickItem': 0}

# {'ESC': 0, 'back': 0, 'drop': 0, 'forward': 0, 'hotbar.1': 0, 'hotbar.2': 0, 'hotbar.3': 0, 'hotbar.4': 0, 'hotbar.5': 0, 'hotbar.6': 0, 'hotbar.7': 0, 'hotbar.8': 0, 'hotbar.9': 0, 
# 'inventory': 0, 'jump': 0, 'left': 0, 'right': 0, 'sneak': 0, 'sprint': 0, 'swapHands': 0, 'camera': array([0, 0]), 'attack': 0, 'use': 0, 'pickItem': 0}

#ACTION_KEYS = [
#    "inventory", "ESC", "hotbar.1", "hotbar.2", "hotbar.3", "hotbar.4", "hotbar.5", "hotbar.6", "hotbar.7", "hotbar.8", "hotbar.9", 
#    "forward", "back", "left", "right", "jump", "sneak", "sprint", "swapHands", "attack", "use", "pickItem", "drop", "cameraX", "cameraY"
#]
ACTION_KEYS = [
    'ESC', 'back', 'drop', 'forward', 'hotbar.1', 'hotbar.2', 'hotbar.3', 'hotbar.4', 'hotbar.5', 'hotbar.6', 'hotbar.7', 'hotbar.8', 'hotbar.9', 
    'inventory', 'jump', 'left', 'right', 'sneak', 'sprint', 'swapHands', 'camera', 'attack', 'use', 'pickItem'
]

def load_actions(action_path):
    return torch.load(action_path, weights_only=False)

def draw_overlay(frame, actions, frame_idx, width, height):
    overlay = frame.copy()
    #print(actions)
    for i, key in enumerate(ACTION_KEYS):
        if key=='camera':
            #print(actions[frame_idx])
            value = actions[frame_idx].get(key, 0)
            #print(value)
            color = (0, 255, 0)
            cv2.putText(overlay, f"x: {value[1]}", (60, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.25, color, 1)
            cv2.putText(overlay, f"y: {value[0]}", (60, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.25, color, 1)
            continue  # Camera movement will be visualized differently
        
        value = actions[frame_idx].get(key, 0)
        #if(value>0):
        #    print(actions[frame_idx])
        color = (0, 255, 0) if value else (255, 0, 0)
        cv2.putText(overlay, f"{key}: {value}", (10, 30 + i * 10), cv2.FONT_HERSHEY_SIMPLEX, 0.25, color, 1)
    
    return overlay

def overlay_video(video_path, action_path, output_path):
    actions = load_actions(action_path)
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error opening video file.")
        return
    # Get frame width and height
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Frame width: {width}")
    print(f"Frame height: {height}")

    actual_fps = cap.get(cv2.CAP_PROP_FPS)  # Get the actual FPS of the input video
    print("actual_fps:",actual_fps)
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
#overlay_video("./recording_data/Player729-f153ac423f61-20210806-224813.chunk_000.mp4", "./recording_data/Player729-f153ac423f61-20210806-224813.chunk_000.actions.pt", "overlay_output/replay_with_overlay.mp4")
#overlay_video("./recording_data/replay_20250406_220923.mp4", "./recording_data/actions_20250406_220923.pt", "overlay_output/replay_with_overlay_test.mp4")
#overlay_video("./recording_data/replay_20250406_212840.mp4", "./recording_data/actions_20250406_212840.pt", "overlay_output/replay_with_overlay_test.mp4")
#overlay_video("./recording_data/replay_20250405_225643.mp4", "./recording_data/actions_20250405_225643.pt", "overlay_output/replay_with_overlay_test.mp4")
#overlay_video("./recording_data/replay_20250407_180354.mp4", "./recording_data/actions_20250407_180354.pt", "overlay_output/replay_with_overlay_test.mp4")
#overlay_video("./recording_data/replay_20250407_181728.mp4", "./recording_data/actions_20250407_181728.pt", "overlay_output/replay_with_overlay_test.mp4")
#overlay_video("./recording_data/replay_20250407_182239.mp4", "./recording_data/actions_20250407_182239.pt", "overlay_output/replay_with_overlay_test.mp4")
overlay_video("./recording_data/replay_20250407_194351.mp4", "./recording_data/actions_20250407_194351.pt", "overlay_output/replay_with_overlay_test.mp4")
