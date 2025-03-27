"""
This is the file that runs during gameplay to record the screen, the keyboard, and the mouse
"""

import cv2
import numpy as np
import torch
import time
import mss
from pynput import keyboard, mouse
from threading import Thread, Lock
from collections import deque
import datetime
from PIL import Image
from find_health_bar_aspect_ratio import count_hearts

# Constants
ACTION_KEYS = [
    "inventory", "ESC", "hotbar.1", "hotbar.2", "hotbar.3", "hotbar.4", "hotbar.5", "hotbar.6", "hotbar.7", "hotbar.8", "hotbar.9", 
    "forward", "back", "left", "right", "cameraX", "cameraY", "jump", "sneak", "sprint", "swapHands", "attack", "use", "pickItem", "drop"
]

KEY_ACTION_MAP = {
    "w": "forward",
    "s": "back",
    "a": "left",
    "d": "right",
    "space": "jump",
    "shift": "sneak",
    "ctrl": "sprint",
    "e": "inventory",
    "esc": "ESC",
    "1": "hotbar.1",
    "2": "hotbar.2",
    "3": "hotbar.3",
    "4": "hotbar.4",
    "5": "hotbar.5",
    "6": "hotbar.6",
    "7": "hotbar.7",
    "8": "hotbar.8",
    "9": "hotbar.9",
    "q": "drop",
    "f": "swapHands",
    "r": "pickItem",
}

last_forward_press_time = 0  # Track time of last 'w' press
double_tap_threshold = 0.3  # 300ms for sprint detection

# Screen recording buffer
FPS = 60
BUFFER_SECONDS = 60
FRAME_BUFFER = deque(maxlen=FPS * BUFFER_SECONDS)
ACTION_BUFFER = deque(maxlen=FPS * BUFFER_SECONDS)  # Stores recent actions
lock = Lock()
recording = False

damage_detected = False
damage_timer = None
timeout_seconds = 5
current_inventory_slot = 1
current_health = 0
def player_taking_damage():
    # returns true if the player took damage
    # also returns true if the player is dead (and wasn't dead the previous frame)
    # we calculate this by seeing whether the updated health is less than the previous health
    # there might be a more efficient way than taking another screenshot (we take 2 screenshots every frame), but idk and if it works it works
    global current_health
    sct = mss.mss()
    monitor = sct.monitors[1]
    img = np.array(sct.grab(monitor))[:, :, :3]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = Image.fromarray(img)
    img.show() 
    updated_health = count_hearts(img)
    # if updated_health>0: print(updated_health)
    took_damage =(updated_health<current_health)
    dead = (updated_health==0 and current_health>0)
    current_health=updated_health
    return took_damage, dead

def record_screen():
    sct = mss.mss()
    monitor = sct.monitors[1]
    while True:
        img = np.array(sct.grab(monitor))[:, :, :3]
        img = cv2.resize(img, (640, 360))
        #img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        with lock:
            FRAME_BUFFER.append(img)
        time.sleep(1 / FPS)


def save_recording():
    with lock:
        frames = list(FRAME_BUFFER)
        saved_actions = list(ACTION_BUFFER)
    
    if not frames or not saved_actions:
        print("No frames/actions to save.")
        return
    
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"replay_{timestamp}.mp4"
    actions_file = f"actions_{timestamp}.pt"
    
    out = cv2.VideoWriter(output_file, fourcc, FPS, (width, height))
    
    for frame in frames:
        out.write(frame)
    out.release()
    print(f"Saved recording to {output_file}")
    # Save Actions
    torch.save(saved_actions, actions_file)
    print(f"Saved actions to {actions_file}")


def monitor_triggers():
    global damage_detected, damage_timer
    while True:
        took_damage, dead = player_taking_damage()
        if dead:
            print("Death detected! Saving recording...")
            save_recording()
            damage_detected = False
            damage_timer = None
        elif took_damage:
            current_time = time.time()
            if not damage_detected:
                print("Damage detected! Initializing recording buffer...")
                damage_detected = True
                damage_end_time = current_time + timeout_seconds
            else:
                # Extend the timeout instead of resetting everything
                damage_end_time = current_time + timeout_seconds
        # Check if enough time has passed since last damage to stop recording
        if damage_detected and damage_end_time and time.time() > damage_end_time:
            print("Saving damage-triggered recording...")
            save_recording()
            damage_detected = False
            damage_end_time = None
        time.sleep(0.1)  # Check every 100ms

def on_press(key):
    global last_forward_press_time
    global current_inventory_slot
    try:
        action = None
        if hasattr(key, 'char') and key.char:
            action = KEY_ACTION_MAP.get(key.char.lower())
        else:
            action = KEY_ACTION_MAP.get(key.name)
        
        if action:
            ACTION_BUFFER.append({"time": time.time(), "action": action, "value": 1})
            
            # Detect sprint (double-tap 'w')
            if action == "forward":
                current_time = time.time()
                if current_time - last_forward_press_time < double_tap_threshold:
                    ACTION_BUFFER.append({"time": current_time, "action": "sprint", "value": 1})
                last_forward_press_time = current_time
            elif action == "hotbar.1":
                current_inventory_slot=1
            elif action == "hotbar.2":
                current_inventory_slot=2
            elif action == "hotbar.3":
                current_inventory_slot=3
            elif action == "hotbar.4":
                current_inventory_slot=4
            elif action == "hotbar.5":
                current_inventory_slot=5
            elif action == "hotbar.6":
                current_inventory_slot=6
            elif action == "hotbar.7":
                current_inventory_slot=7
            elif action == "hotbar.8":
                current_inventory_slot=8
            elif action == "hotbar.9":
                current_inventory_slot=9
    except AttributeError:
        pass


def on_release(key):
    action = None
    if hasattr(key, 'char') and key.char:
        action = KEY_ACTION_MAP.get(key.char.lower())
    else:
        action = KEY_ACTION_MAP.get(key.name)
    
    if action:
        ACTION_BUFFER.append({"time": time.time(), "action": action, "value": 0})


def on_click(x, y, button, pressed):
    action = "attack" if button == mouse.Button.left else "use"
    ACTION_BUFFER.append({"time": time.time(), "action": action, "value": int(pressed)})

def on_move(x, y):
    ACTION_BUFFER.append({"time": time.time(), "action": "cameraX", "value": x})
    ACTION_BUFFER.append({"time": time.time(), "action": "cameraY", "value": y})

def on_scroll(x, y, dx, dy):
    global current_inventory_slot
    if dy > 0:  # Scroll up
        current_inventory_slot = (current_inventory_slot + 1) % 9  # Wrap around at 9 slots
    elif dy < 0:  # Scroll down
        current_inventory_slot = (current_inventory_slot - 1) % 9
    ACTION_BUFFER.append({"time": time.time(), "action": "hotbar."+current_inventory_slot,"value":1})


def main():
    output_actions = "user.actions.pt"
    print("running")
    # Start screen recording buffer
    screen_thread = Thread(target=record_screen, daemon=True)
    screen_thread.start()
    
    # Start monitoring triggers
    trigger_thread = Thread(target=monitor_triggers, daemon=True)
    trigger_thread.start()
    
    # Start user input tracking
    with keyboard.Listener(on_press=on_press, on_release=on_release) as k_listener, \
         mouse.Listener(on_click=on_click, on_move=on_move, on_scroll=on_scroll) as m_listener:
        k_listener.join()
        m_listener.join()
    print("Action logging completed.")
    

if __name__ == "__main__":
    main()
