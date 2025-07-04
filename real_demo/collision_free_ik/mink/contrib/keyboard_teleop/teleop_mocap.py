from functools import partial

import mujoco
import numpy as np

from . import keycodes


class TeleopMocap:
    """
    Class to handle keyboard input for teleoperation.
    The class provides methods to toggle teleoperation on/off,
    switch between manual and non-manual modes,
    adjust step sizes / speed for movement and rotation,
    and select movements based on key presses.
    """

    def __init__(self, data):
        self.on = False
        self.data = data
        self.reset_state()
        self.actions = {
            keycodes.KEY_N: self.toggle_manual,  # n: toggle non-manual mode
            keycodes.KEY_PERIOD: self.toggle_rotation,  # .: toggle rotation mode
            keycodes.KEY_8: self.toggle_mocap,  # 8: toggle mocap data
            keycodes.KEY_EQUAL: partial(self.toggle_speed, 1),  # =/+: increase speed
            keycodes.KEY_MINUS: partial(self.toggle_speed, -1),  # -: decrease speed
            keycodes.KEY_UP: partial(
                self.movement_select, keycodes.KEY_UP, 0, 1
            ),  # Up arrow
            keycodes.KEY_DOWN: partial(
                self.movement_select, keycodes.KEY_DOWN, 0, -1
            ),  # Down arrow
            keycodes.KEY_RIGHT: partial(
                self.movement_select, keycodes.KEY_RIGHT, 1, 1
            ),  # Right arrow
            keycodes.KEY_LEFT: partial(
                self.movement_select, keycodes.KEY_LEFT, 1, -1
            ),  # Left arrow
            keycodes.KEY_7: partial(self.movement_select, keycodes.KEY_7, 2, 1),  # 6
            keycodes.KEY_6: partial(self.movement_select, keycodes.KEY_6, 2, -1),  # 7
        }
        self.movements = {
            keycodes.KEY_UP: partial(self.movement_select, -1, 0, 1),  # Up arrow
            keycodes.KEY_DOWN: partial(self.movement_select, -1, 0, -1),  # Down arrow
            keycodes.KEY_RIGHT: partial(self.movement_select, -1, 1, 1),  # Right arrow
            keycodes.KEY_LEFT: partial(self.movement_select, -1, 1, -1),  # Left arrow
            keycodes.KEY_7: partial(self.movement_select, -1, 2, 1),  # 6
            keycodes.KEY_6: partial(self.movement_select, -1, 2, -1),  # 7
        }
        self.opposite_keys = {
            keycodes.KEY_UP: keycodes.KEY_DOWN,
            keycodes.KEY_DOWN: keycodes.KEY_UP,
            keycodes.KEY_RIGHT: keycodes.KEY_LEFT,
            keycodes.KEY_LEFT: keycodes.KEY_RIGHT,
            keycodes.KEY_7: keycodes.KEY_6,
            keycodes.KEY_6: keycodes.KEY_7,
        }

    def __call__(self, key):
        # Toggle teleop on/off
        if key == keycodes.KEY_9:  # 9
            self.toggle_on()
            return

        # Do nothing if teleop is off
        if not self.on:
            return

        if key in self.actions:
            self.actions[key]()

    def auto_key_move(self):
        """
        Automatically move the mocap body based on key presses.
        """

        if not self.on:
            return

        for key, action in self.movements.items():
            if self.keys[key]:
                action()

    def movement_select(self, key, axis, direction):
        """
        Select the movement direction based on the key pressed.
        """

        if not self.manual and key != -1:
            self.keys[key] = not self.keys[key]
            self.keys[self.opposite_keys[key]] = False
        elif not self.manual and key == -1:
            self.rot_or_trans(key, axis, direction)
        elif self.manual:
            self.rot_or_trans(key, axis, direction)

    def rot_or_trans(self, key, axis, direction):
        """
        Adjust the position or rotation of the mocap body based on rotation mode.
        """

        if self.rotation:
            self.adjust_rotation(key, axis, direction)
        else:
            self.adjust_position(key, axis, direction)

    def adjust_position(self, key, axis, direction):
        """
        Adjust the position of the mocap body in the specified direction
        based on the axis and step size.
        """

        q = self.data.mocap_quat[self.mocap_idx].copy()
        unit_vec = self.unit_vec_from_quat(q, axis)
        step_size = self.m_step_size if self.manual else self.nm_step_size
        self.data.mocap_pos[self.mocap_idx] += direction * step_size * unit_vec

    def adjust_rotation(self, key, axis, direction):
        """
        Adjust the rotation of the mocap body in the specified direction
        based on the axis and step size.
        """

        q = self.data.mocap_quat[self.mocap_idx].copy()
        unit_vec = self.unit_vec_from_quat(q, axis)

        # Rotate the quaternion by the specified angle around the axis.
        quat_rot = np.zeros(shape=(4,), dtype=np.float64)
        result = np.zeros(shape=(4,), dtype=np.float64)
        step_size = self.m_rotation_step if self.manual else self.nm_rotation_step
        angle = direction * step_size
        angle_rad = np.deg2rad(angle)
        unit_vec = unit_vec / np.linalg.norm(unit_vec)
        mujoco.mju_axisAngle2Quat(quat_rot, unit_vec, angle_rad)
        mujoco.mju_mulQuat(result, quat_rot, q)

        self.data.mocap_quat[self.mocap_idx] = result

    def unit_vec_from_quat(self, q, axis):
        """
        Compute the unit vector corresponding to the specified axis
        from the given quaternion.
        """

        rot = np.zeros(shape=(9,), dtype=np.float64)
        mujoco.mju_quat2Mat(rot, q)
        rot = rot.reshape((3, 3))
        unit_vec = rot[:, axis]

        return unit_vec

    def toggle_on(self):
        self.on = not self.on
        state = "On" if self.on else "Off"
        print(f"Keyboard Teleoperation toggled: {state}!")
        self.reset_state()
        print()

    def toggle_manual(self):
        self.manual = not self.manual
        manual_state = "On" if self.manual else "Off"
        print(f"Manual mode toggled: {manual_state}!")
        self.reset_keys()
        print()

    def toggle_rotation(self):
        self.rotation = not self.rotation
        state = "On" if self.rotation else "Off"
        print(f"Rotation mode toggled: {state}!")
        self.reset_keys()
        print()

    def toggle_speed(self, direction):
        factor = 1.10 if direction == 1 else 0.9
        if self.manual:
            if self.rotation:
                self.m_rotation_step *= factor
            else:
                self.m_step_size *= factor
        else:
            if self.rotation:
                self.nm_rotation_step *= factor
            else:
                self.nm_step_size *= factor

        output = "Manual" if self.manual else "Non-manual"
        mode = "Rotation" if self.rotation else "Translation"
        if self.manual:
            step_size = self.m_rotation_step if self.rotation else self.m_step_size
        else:
            step_size = self.nm_rotation_step if self.rotation else self.nm_step_size
        print(f"{output} {mode} step size: {step_size:.8f}")

    def toggle_mocap(self):
        self.mocap_idx = (self.mocap_idx + 1) % self.data.mocap_pos.shape[
            0
        ]  # cycle through mocap data
        print(f"Current mocap index: {self.mocap_idx}")

    def reset_keys(self):
        self.keys = {
            keycodes.KEY_UP: False,
            keycodes.KEY_DOWN: False,
            keycodes.KEY_RIGHT: False,
            keycodes.KEY_LEFT: False,
            keycodes.KEY_7: False,
            keycodes.KEY_6: False,
        }

    def reset_step_size(self):
        self.m_step_size = 0.01  # manual step size
        self.m_rotation_step = 10  # manual rotation step
        self.nm_step_size = 1e-4  # non-manual step size
        self.nm_rotation_step = 5e-2  # non-manual rotation step
        print("Step sizes have been reset!")

    def reset_state(self):
        self.reset_keys()
        self.reset_step_size()
        self.manual = True
        self.rotation = False
        self.mocap_idx = 0
        str = f"States have been reset: \n \
        - Manual mode: {self.manual} \
        - Rotation mode: {self.rotation} \n \
        - Mocap index: {self.mocap_idx}"

        print(str)
