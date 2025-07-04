# Keyboard Controls for Teleoperation
This document explains the keyboard input handling system implemented in the `TeleopMocap` class for teleoperation.  
The system allows users to toggle different movement modes, adjust speeds, and control the movement of a mocap body in a simulation.

---

## Overview
The `TeleopMocap` class provides keyboard-based control for moving and rotating the mocap body in a Mujoco simulation.  
Features:
- 6 degrees of freedom (DOF) movement
- Toggling between manual and non-manual movement
- Switching between rotation and translation
- Adjusting movement and rotation step sizes
- Alternating between different mocap data

---

## Key Mappings
| Key | Action |
|-----|--------|
| `9` | Toggle teleoperation On/Off. |
| `n` | Toggle between manual and non-manual mode. |
| `.` | Toggle between rotation and translation mode. |
| `8` | Cycle through different mocap data. |
| `+` | Increase movement step size or movement speed. |
| `-` | Decrease movement step size or movement speed. |
| **Arrow Keys** | **Move (rotation / translation) along the X, Y, and Z axes** |
| `Up` | Move forward (+X) or rotates around X-axis in positive direction. |
| `Down` | Move backward (-X) or rotates around X-axis in negative direction. |
| `Right` | Move right (+Y) or rotates around Y-axis in positive direction. |
| `Left` | Move left (-Y) or rotates around Y-axis in negative direction. |
| `7` | Move up (+Z) or rotates around Z-axis in positive direction. |
| `6` | Move down (-Z) or rotates around Z-axis in negative direction. |

---

## Modes
### **Manual vs. Non-Manual Mode:**
- **Manual Mode**: Iterative movement using arrow keys.
- **Non-Manual Mode**: Continuous movement using arrow keys (to stop continuous movement, re-click the arrow key).

### **Rotation vs. Translation Mode:**
- **Rotation Mode**: Rotation around an axis.
- **Translation Mode**: Movement along an axis.

---

## Example Usage
To use the `TeleopMocap` class, instantiate it and pass the `mjData`.  
Pass the `key_callback_data` method as a `key_callback` in the mujoco viewer.  
Call the `auto_key_move()` in the viewer loop.

```python
import mink
...
data = MjData(model)
...
# Initialize the key callback handler
key_callback = mink.TeleopMocap(data)

# Pass the key callback function into the viewer
with mujoco.viewer.launch_passive(
    model=model, data=data, 
    show_left_ui=False, show_right_ui=False, 
    key_callback=key_callback.key_callback_data
) as viewer:
    while viewer.is_running():
        ...
        key_callback.auto_key_move()
        ...
```

---

## Limitations
`Mink` uses the `mujoco.viewer.launch_passive()` for visualization and maintaining the simulation loop.  
To pass keyboard callbacks, we have to pass a callback function `key_callback(key)` as an argument to the `launch_passive()`.  
However, this has several limitations:
- Only being able to register one key press at a time.
- It can only register one key action (which is the action PRESS); can't register key holds (act.HOLD) or releases (act.RELEASE).
- Doesn't support key + modifier combinations in (e.g., Ctrl + Arrow keys).
- Viewer has a lot of default keybinds, which limits the amount of free keys to use for movement.
