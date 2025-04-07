"""
This is the file that runs during gameplay to record the screen, the keyboard, and the mouse
"""
import cv2
import numpy as np
import torch
import time
import mss
from pynput import keyboard, mouse
from threading import Thread, Lock, Event
from collections import deque
import datetime
from PIL import Image
from find_health_bar_aspect_ratio import count_hearts
import copy
from pathlib import Path

# Constants
ACTION_KEYS = [
    "ESC", "back", "drop", "forward", "hotbar.1", "hotbar.2", "hotbar.3", "hotbar.4", "hotbar.5", "hotbar.6", "hotbar.7", "hotbar.8", "hotbar.9", 
    "inventory", "jump", "left", "right", "sneak", "sprint", "swapHands", "camera", "attack", "use", "pickItem"
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

SPECIAL_KEY_ACTION_MAP = {
    keyboard.Key.space: "jump",
    keyboard.Key.shift: "sneak",
    #keyboard.Key.shift_r: "Right Shift",
    keyboard.Key.ctrl: "sprint",
    #keyboard.Key.ctrl_r: "Right Ctrl",
    keyboard.Key.esc: "ESC"
}

last_forward_press_time = 0  # Track time of last 'w' press
double_tap_threshold = 0.3  # 300ms for sprint detection

# Screen recording buffer
FPS = 20
BUFFER_SECONDS = 15
FRAME_BUFFER = deque(maxlen=FPS * BUFFER_SECONDS)
ACTIONS_IN_A_SINGLE_FRAME = {}
last_frame = None
ACTION_BUFFER = deque(maxlen=FPS * BUFFER_SECONDS)  # Stores recent actions
lock = Lock()

stop_event = Event()  # Global active signal

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
    updated_health = count_hearts(img)
    took_damage =(updated_health<current_health)
    dead = (updated_health==0 and current_health>0)
    current_health=updated_health
    return took_damage, dead

def compile_single_frame_actions():
    global ACTIONS_IN_A_SINGLE_FRAME
    global last_frame
    if not last_frame:
        last_frame = {"ESC":0, "back":0, "drop":0, "forward":0, "hotbar.1":0, "hotbar.2":0, "hotbar.3":0, "hotbar.4":0, "hotbar.5":0, "hotbar.6":0, "hotbar.7":0, "hotbar.8":0, "hotbar.9":0, 
                        "inventory":0, "jump":0, "left":0, "right":0, "sneak":0, "sprint":0, "swapHands":0, "camera":np.array([40,40]), "attack":0, "use":0, "pickItem":0 }
    
    for action in ACTIONS_IN_A_SINGLE_FRAME:
        if "hotbar" in action: continue
        last_frame[action]=ACTIONS_IN_A_SINGLE_FRAME[action]
    ACTIONS_IN_A_SINGLE_FRAME = {}
    return copy.deepcopy(last_frame)

def compress_mouse(dx):
    """
    Stolen from https://github.com/etched-ai/open-oasis/issues/9
    Results in camera coordinates centered on [40,40]
    Note that Oasis's dataset clearly contains camera values such as 39 and 41, which means dx must contain sub-pixel quantities (between 0 and 0.25)
    We'll just assume sub-pixel quantities of 0.25 for now
    """
    #Convert dx to sub-pixel quantity of 0.25
    dx/=4.0

    max_val = 20
    bin_size = 0.5
    mu = 2.7

    dx = np.clip(dx, -max_val, max_val)
    dx /= max_val
    v_encode = np.sign(dx) * (np.log(1.0 + mu * np.abs(dx)) / np.log(1.0 + mu))
    v_encode *= max_val
    dx = v_encode

    return np.round((dx + max_val) / bin_size).astype(np.int64)

def record_screen():
    global ACTIONS_IN_A_SINGLE_FRAME
    global current_x, current_y, dx, dy
    sct = mss.mss()
    monitor = sct.monitors[1]
    dx, dy, current_x, current_y = 0, 0, 0, 0
    last_time = time.time()  # Track time manually
    while not stop_event.is_set():
        current_time = time.time()
        elapsed_time = current_time - last_time

        if elapsed_time >= 1 / (FPS):  # Only capture if enough time has passed
            last_time = current_time  # Update last captured frame time
            img = np.array(sct.grab(monitor)) #This line of code may take longer than 1/20th of a second :(
            img = img[:, :, :3]
            img = cv2.resize(img, (640, 360)) #This line of code also takes a while
            
            # Up results in small-y
            # Left results in small-x
            # Note that due to a quirk in the VPT data, x and y are swapped so it looks like (y, x)
            ACTIONS_IN_A_SINGLE_FRAME["camera"] = np.array([compress_mouse(dy), compress_mouse(dx)])
            current_x, current_y = mouse.Controller().position
            dx, dy = 0, 0
            action_in_a_single_frame = compile_single_frame_actions()
            with lock:
                FRAME_BUFFER.append(img)
                ACTION_BUFFER.append(action_in_a_single_frame)


def save_recording():
    global FRAME_BUFFER
    global ACTION_BUFFER
    with lock:
        frames = list(FRAME_BUFFER)
        saved_actions = list(ACTION_BUFFER)
    
    if not frames or not saved_actions:
        print("No frames/actions to save.")
        return

    # Get the path to the Downloads folder and create the target directory
    downloads_path = Path.home() / "Downloads"
    save_dir = downloads_path / "minecraft_recording_data"
    save_dir.mkdir(parents=True, exist_ok=True)  # Create folder if it doesn't exist

    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = save_dir / f"replay_{timestamp}.mp4"
    actions_file = save_dir / f"actions_{timestamp}.pt"

    # Save video
    out = cv2.VideoWriter(str(output_file), fourcc, FPS, (width, height))
    for frame in frames:
        frame = frame[:, :, :3]
        out.write(frame)
    out.release()
    print(f"Saved recording to {output_file}")

    # Save actions
    torch.save(saved_actions, str(actions_file))
    print(f"Saved actions to {actions_file}")

    # Clear buffers
    FRAME_BUFFER = deque(maxlen=FPS * BUFFER_SECONDS)
    ACTION_BUFFER = deque(maxlen=FPS * BUFFER_SECONDS)


def monitor_triggers():
    global damage_detected, damage_timer
    while not stop_event.is_set():
        # Make sure the triggers don't occur if we don't have a full frame buffer
        if len(FRAME_BUFFER)!=FRAME_BUFFER.maxlen: 
            time.sleep(1/FPS)
            continue
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
        time.sleep(1/FPS)  # Check every 100ms

def on_press(key):
    global last_forward_press_time
    global current_inventory_slot
    global screen_thread
    global trigger_thread
    global k_listener
    global m_listener
    try:
        if key.char == 'l':
            #key to stop the program
            stop_event.set()
            print("Stopping recording")
            screen_thread.join()
            trigger_thread.join()
            k_listener.stop()
            m_listener.stop()
            print("All threads stopped.")
        action = None
        if hasattr(key, 'char') and key.char:
            action = KEY_ACTION_MAP.get(key.char.lower())
        else:
            action = KEY_ACTION_MAP.get(key.name)
        
        if action:
            #ACTION_BUFFER.append({action: 1})
            ACTIONS_IN_A_SINGLE_FRAME[action] = 1
            
            # Detect sprint (double-tap 'w')
            if action == "forward":
                if last_frame["sprint"]==1 or time.time() - last_forward_press_time < double_tap_threshold:
                    #ACTION_BUFFER.append({"sprint": 1})
                    ACTIONS_IN_A_SINGLE_FRAME["sprint"] = 1
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
        action = SPECIAL_KEY_ACTION_MAP.get(key)
        if action:
            #ACTION_BUFFER.append({action: 1})
            ACTIONS_IN_A_SINGLE_FRAME[action] = 1


def on_release(key):
    global last_forward_press_time
    action = None
    if hasattr(key, 'char') and key.char:
        action = KEY_ACTION_MAP.get(key.char.lower())
    else:
        action = KEY_ACTION_MAP.get(key.name)
    
    if action:
        #ACTION_BUFFER.append({action: 0})
        ACTIONS_IN_A_SINGLE_FRAME[action] = 0
        if action == "forward":
            ACTIONS_IN_A_SINGLE_FRAME["sprint"] = 0
            last_forward_press_time = time.time()


def on_click(x, y, button, pressed):
    action = "attack" if button == mouse.Button.left else "use"
    #ACTION_BUFFER.append({action: int(pressed)})
    ACTIONS_IN_A_SINGLE_FRAME[action] = int(pressed)

def on_move(x, y):
    """
    Note that when not in a menu (such as your inventory or the pause menu), 
    Minecraft will constantly reset your mouse location to the last location it was in before you started playing

    As a result, we can't just find (dx, dy) over 20 fps by just taking the difference between 2 mouse locations (since Minecraft runs at faster than 20 fps)
    We need to accumulate (dx, dy) over potentially multiple updates across 1 frame @ 20 fps
    """
    global current_x, current_y, dx, dy
    dx += x-current_x
    dy += y-current_y
    

def on_scroll(x, y, dx, dy):
    """
    I currently don't have a good solution for initializing current_inventory_slot
    It's possible to spawn in with the current inventory slot not being the first slot (ie: loading a save where you're current inventory slot was your 2nd slot)
    It's encouraged that when your inputs are recording that you start by pressing a key 1-9 to 'anchor' current_inventory slot before scrolling
    """
    global current_inventory_slot
    if dy > 0:  # Scroll up
        current_inventory_slot = (current_inventory_slot - 1) % 9  # Wrap around at 9 slots
    elif dy < 0:  # Scroll down
        current_inventory_slot = (current_inventory_slot + 1) % 9
    ACTIONS_IN_A_SINGLE_FRAME["hotbar."+str(current_inventory_slot+1)] = 1
    for i in range(9):
        if i==current_inventory_slot: continue
        ACTIONS_IN_A_SINGLE_FRAME["hotbar."+str(i+1)] = 0


def main():
    global screen_thread
    global trigger_thread
    global k_listener
    global m_listener
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
    

if __name__ == "__main__":
    main()
