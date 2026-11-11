import os
import select
import sys
from dataclasses import dataclass, field

import numpy as np


def axis_value(value, deadzone):
    if not np.isfinite(value):
        return 0.0
    value = float(np.clip(value, -1, 1))
    return float(np.sign(value) * max(0, abs(value) - deadzone) / (1 - deadzone))


def joystick_velocity(axes, bounds, deadzone):
    normalized = np.array([axis_value(value, deadzone) for value in axes])
    # An axis with no trained reverse range remains zero on that half of the stick.
    return np.where(
        normalized >= 0,
        normalized * np.maximum(bounds[:, 1], 0),
        -normalized * np.minimum(bounds[:, 0], 0),
    )


@dataclass
class Controls:
    bounds: np.ndarray
    target: np.ndarray = field(default_factory=lambda: np.zeros(3))
    command: np.ndarray = field(default_factory=lambda: np.zeros(3))
    paused: bool = False
    reset_requested: bool = False
    quit: bool = False
    joystick_inhibited: bool = False
    last_action: str = "READY"

    def stop(self, *, inhibit=True):
        self.target[:] = self.command[:] = 0
        self.joystick_inhibited = inhibit

    def key(self, key, *, joystick=False):
        key = key.lower()
        if key in ("\x1b", "\x03"):
            self.quit = True
        elif key == " ":
            self.stop()
            self.last_action = "ZERO COMMAND"
        elif key == "p":
            self.paused = not self.paused
            self.stop(inhibit=self.paused)
            self.last_action = "PAUSED" if self.paused else "RESUMED"
        elif key == "r":
            self.reset_requested = True
            self.paused = False
            self.stop(inhibit=False)
            self.last_action = "RESET"
        elif not joystick and key in "wsadqe":
            axis, sign = {"w": (0, 1), "s": (0, -1), "a": (1, 1), "d": (1, -1),
                          "q": (2, 1), "e": (2, -1)}[key]
            value = self.target[axis] + sign * 0.1
            self.target[axis] = np.clip(value, min(0, self.bounds[axis, 0]), max(0, self.bounds[axis, 1]))

    def joystick(self, velocity, enabled):
        if not enabled:
            self.stop()
            self.joystick_inhibited = False
        elif self.joystick_inhibited:
            if not np.any(velocity):
                self.joystick_inhibited = False
        elif not self.paused:
            self.target[:] = velocity

    def advance(self, dt, acceleration):
        if self.paused:
            self.command[:] = 0
        else:
            self.command += np.clip(self.target - self.command, -acceleration * dt, acceleration * dt)
        return self.command.copy()


class Keyboard:
    def __init__(self):
        self.settings = None
        self.fd = None

    def __enter__(self):
        if sys.stdin.isatty() and os.name != "nt":
            import termios
            import tty

            self.fd = sys.stdin.fileno()
            self.settings = termios.tcgetattr(self.fd)
            tty.setcbreak(self.fd)
        return self

    def poll(self):
        if not sys.stdin.isatty():
            return []
        if os.name == "nt":
            import msvcrt

            keys = []
            while msvcrt.kbhit():
                keys.append(msvcrt.getwch())
            return keys
        if select.select([self.fd], [], [], 0)[0]:
            return list(os.read(self.fd, 128).decode(errors="ignore"))
        return []

    def __exit__(self, *_):
        if self.settings is not None:
            import termios

            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.settings)


class Joystick:
    def __init__(self, index, axes=None, signs=(-1, -1, -1), deadzone=0.12, deadman=-1,
                 *, reset_button=None, stop_button=None, pause_button=None):
        os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
        os.environ.setdefault("SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS", "1")
        # mjpython runs Python off the Cocoa main thread; SDL only handles gamepads here.
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        import pygame
        from pygame._sdl2 import controller

        self.pg = pygame
        self.controller_api = controller
        self.controller = None
        self.device = None
        self.index, self.axes, self.signs = index, axes, np.asarray(signs)
        self.deadzone, self.deadman = deadzone, deadman
        self.button_overrides = {"r": reset_button, " ": stop_button, "p": pause_button}
        self.bindings = {}
        self._pressed = set()
        self.enabled = False
        self.velocity = np.zeros(3)
        self.name = "disconnected"
        self.instance = None
        self._owns_display = not pygame.display.get_init()
        self._owns_joystick = not pygame.joystick.get_init()
        self._owns_controller = not controller.get_init()
        pygame.display.init()
        pygame.joystick.init()
        controller.init()
        try:
            self._connect()
        except BaseException:
            self.close()
            raise

    def _connect(self):
        if self.pg.joystick.get_count() <= self.index:
            return
        device = self.pg.joystick.Joystick(self.index)
        self.device = device
        self.instance = device.get_instance_id()
        self.name = device.get_name()
        if self.controller_api.is_controller(self.index):
            self.controller = self.controller_api.Controller(self.index)
        self.axis_ids = self.axes
        if self.axis_ids is None:
            self.axis_ids = (
                [self.pg.CONTROLLER_AXIS_LEFTY, self.pg.CONTROLLER_AXIS_LEFTX, self.pg.CONTROLLER_AXIS_RIGHTX]
                if self.controller else [1, 0, 2]
            )
        if (self.axes is not None or self.controller is None) and max(self.axis_ids) >= device.get_numaxes():
            raise ValueError("Joystick mapping exceeds device axes; set --axes")
        defaults = {
            "r": (self.pg.CONTROLLER_BUTTON_A, "a", 0),
            " ": (self.pg.CONTROLLER_BUTTON_B, "b", 1),
            "p": (self.pg.CONTROLLER_BUTTON_START, "start", 7),
            "enable": (self.pg.CONTROLLER_BUTTON_LEFTSHOULDER, "leftshoulder", 4),
        }
        mapping = self.controller.get_mapping() if self.controller else {}
        physical = set()
        self.bindings = {}
        for key, (semantic, name, fallback) in defaults.items():
            override = self.deadman if key == "enable" else self.button_overrides[key]
            if override == -1:
                self.bindings[key] = None
                continue
            if override is None and self.controller:
                if name not in mapping:
                    raise ValueError(f"Controller has no {name} mapping; configure its button index explicitly")
                binding, physical_button = ("controller", semantic), mapping[name]
            else:
                index = fallback if override is None else override
                if not 0 <= index < device.get_numbuttons():
                    raise ValueError(f"Button {index} exceeds device buttons; configure --*-button options")
                binding, physical_button = ("joystick", index), f"b{index}"
            if physical_button in physical:
                raise ValueError("Enable, reset, stop and pause must use different buttons")
            physical.add(physical_button)
            self.bindings[key] = binding
        # A button held while connecting must be released before it can trigger an action.
        self._pressed = {key for key in self.button_overrides if self._button(self.bindings[key])}

    def _button(self, binding):
        if binding is None:
            return False
        source, index = binding
        return bool((self.controller if source == "controller" else self.device).get_button(index))

    def _edge(self, key, down, keys):
        if down:
            if key not in self._pressed:
                keys.append(key)
                self._pressed.add(key)
        else:
            self._pressed.discard(key)

    def _disconnect(self):
        if self.controller is not None:
            self.controller.quit()
            self.controller = None
        if self.device is not None:
            self.device.quit()
            self.device = None
        self.instance = None
        self.name = "disconnected"
        self.enabled = False
        self.velocity[:] = 0
        self._pressed.clear()

    def poll(self, bounds):
        keys = []
        button_events = {
            self.pg.JOYBUTTONDOWN: ("joystick", True),
            self.pg.JOYBUTTONUP: ("joystick", False),
            self.pg.CONTROLLERBUTTONDOWN: ("controller", True),
            self.pg.CONTROLLERBUTTONUP: ("controller", False),
        }
        for event in self.pg.event.get():
            if event.type == self.pg.JOYDEVICEREMOVED and event.instance_id == self.instance:
                self._disconnect()
                keys.clear()
            elif event.type == self.pg.JOYDEVICEADDED and self.device is None:
                self._connect()
            elif event.type in button_events and event.instance_id == self.instance:
                source, down = button_events[event.type]
                for key in self.button_overrides:
                    if self.bindings[key] == (source, event.button):
                        self._edge(key, down, keys)
        if self.device is None:
            return np.zeros(3), False, keys
        for key in self.button_overrides:
            self._edge(key, self._button(self.bindings[key]), keys)
        if self.controller and self.axes is None:
            axes = np.array([self.controller.get_axis(index) for index in self.axis_ids]) / 32767
        else:
            axes = np.array([self.device.get_axis(index) for index in self.axis_ids])
        binding = self.bindings["enable"]
        self.enabled = binding is None or self._button(binding)
        self.velocity = joystick_velocity(axes * self.signs, bounds, self.deadzone)
        return self.velocity.copy(), self.enabled, keys

    def status(self, controls):
        if self.device is None:
            return "DISCONNECTED"
        if controls.paused:
            return "PAUSED"
        if not self.enabled:
            return "HOLD LB" if self.deadman is None and self.controller else "HOLD ENABLE BUTTON"
        if controls.joystick_inhibited:
            return "CENTER STICKS" if self.bindings["enable"] is None else "CENTER STICKS OR RELEASE ENABLE"
        return "ACTIVE"

    def close(self):
        self._disconnect()
        if self._owns_controller:
            self.controller_api.quit()
        if self._owns_joystick:
            self.pg.joystick.quit()
        if self._owns_display:
            self.pg.display.quit()
