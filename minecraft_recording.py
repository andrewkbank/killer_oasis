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
from PIL import ImageGrab
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

actions = []  # Stores user actions
last_forward_press_time = 0  # Track time of last 'w' press
double_tap_threshold = 0.3  # 300ms for sprint detection

# Screen recording buffer
FPS = 30
BUFFER_SECONDS = 60
FRAME_BUFFER = deque(maxlen=FPS * BUFFER_SECONDS)
lock = Lock()
recording = False

damage_detected = False
damage_timer = None
timeout_seconds = 5

sct = mss.mss()
monitor = sct.monitors[1]
current_inventory_slot = 1
current_health = -1
def player_taking_damage():
    # returns true if the player took damage
    # we calculate this by seeing whether the updated health is less than the previous health
    # there might be a more efficient way than taking another screenshot (we take 2 screenshots every frame), but idk and if it works it works
    img = np.array(sct.grab(monitor))[:, :, :3]
    updated_health = count_hearts(img)

    took_damage =(updated_health<current_health)
    current_health=updated_health
    return took_damage

def player_death():
    # returns true if the player is dead
    return (current_health==0)

def record_screen():
    while True:
        img = np.array(sct.grab(monitor))[:, :, :3]
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        with lock:
            FRAME_BUFFER.append(img)
        time.sleep(1 / FPS)


def save_recording():
    with lock:
        frames = list(FRAME_BUFFER)
    
    if not frames:
        print("No frames to save.")
        return
    
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"replay_{timestamp}.mp4"
    
    out = cv2.VideoWriter(output_file, fourcc, FPS, (width, height))
    
    for frame in frames:
        out.write(frame)
    out.release()
    print(f"Saved recording to {output_file}")


def monitor_triggers():
    global damage_detected, damage_timer
    while True:
        if player_death():
            print("Death detected! Saving recording...")
            save_recording()
            damage_detected = False
            damage_timer = None
        elif player_taking_damage():
            if not damage_detected:
                print("Damage detected! Initializing recording buffer...")
                damage_detected = True
            
            if damage_timer is not None:
                damage_timer.cancel()
            
            damage_timer = Thread(target=delayed_save)
            damage_timer.start()
        time.sleep(0.1)  # Check every 100ms


def delayed_save():
    global damage_detected, damage_timer
    time.sleep(timeout_seconds)
    print("Saving damage-triggered recording...")
    save_recording()
    damage_detected = False
    damage_timer = None


def on_press(key):
    global last_forward_press_time
    try:
        action = None
        if hasattr(key, 'char') and key.char:
            action = KEY_ACTION_MAP.get(key.char.lower())
        else:
            action = KEY_ACTION_MAP.get(key.name)
        
        if action:
            actions.append({"time": time.time(), "action": action, "value": 1})
            
            # Detect sprint (double-tap 'w')
            if action == "forward":
                current_time = time.time()
                if current_time - last_forward_press_time < double_tap_threshold:
                    actions.append({"time": current_time, "action": "sprint", "value": 1})
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
        actions.append({"time": time.time(), "action": action, "value": 0})


def on_click(x, y, button, pressed):
    action = "attack" if button == mouse.Button.left else "use"
    actions.append({"time": time.time(), "action": action, "value": int(pressed)})

def on_move(x, y):
    actions.append({"time": time.time(), "action": "cameraX", "value": x})
    actions.append({"time": time.time(), "action": "cameraY", "value": y})

def on_scroll(x, y, dx, dy):
    if dy > 0:  # Scroll up
        current_inventory_slot = (current_inventory_slot + 1) % 9  # Wrap around at 9 slots
    elif dy < 0:  # Scroll down
        current_inventory_slot = (current_inventory_slot - 1) % 9

def save_actions(path):
    torch.save(actions, path)


def main():
    output_actions = "user.actions.pt"
    
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
    
    save_actions(output_actions)
    print("Action logging completed.")
    

if __name__ == "__main__":
    main()
