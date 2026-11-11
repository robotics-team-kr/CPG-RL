import ctypes
from pathlib import Path

import numpy as np
import pytest
import torch

from cpg_console.cli import advance_simulation, parse_args
from cpg_console.controls import Controls, Joystick
from cpg_console.runtime import Runtime, find_checkpoint

BOUNDS = np.array([[-0.5, 0.8], [-0.3, 0.3], [-0.6, 0.6]])


@pytest.fixture
def gamepad(monkeypatch):
    monkeypatch.setenv("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    import pygame
    from pygame._sdl2 import controller

    sdl = ctypes.CDLL(pygame.base.__file__)
    if not hasattr(sdl, "SDL_JoystickAttachVirtual"):
        pytest.skip("SDL virtual joystick API is not exported on this platform")
    for name, arguments, result in (
        ("SDL_JoystickAttachVirtual", [ctypes.c_int] * 4, ctypes.c_int),
        ("SDL_JoystickOpen", [ctypes.c_int], ctypes.c_void_p),
        ("SDL_JoystickSetVirtualAxis", [ctypes.c_void_p, ctypes.c_int, ctypes.c_int16], ctypes.c_int),
        ("SDL_JoystickSetVirtualButton", [ctypes.c_void_p, ctypes.c_int, ctypes.c_uint8], ctypes.c_int),
        ("SDL_JoystickClose", [ctypes.c_void_p], None),
        ("SDL_JoystickDetachVirtual", [ctypes.c_int], ctypes.c_int),
        ("SDL_GameControllerAddMapping", [ctypes.c_char_p], ctypes.c_int),
    ):
        function = getattr(sdl, name)
        function.argtypes, function.restype = arguments, result
    pygame.display.init()
    pygame.joystick.init()
    controller.init()
    index = sdl.SDL_JoystickAttachVirtual(1, 6, 16, 0)
    assert index >= 0
    pointer = sdl.SDL_JoystickOpen(index)
    assert pointer
    raw = pygame.joystick.Joystick(index)
    # Xbox Series X's observed macOS mapping: LB=9, Start=6, stick click=7.
    mapping = (
        f"{raw.get_guid()},CPG Test Controller,a:b0,b:b1,x:b2,y:b3,back:b4,guide:b5,start:b6,"
        "leftstick:b7,rightstick:b8,leftshoulder:b9,rightshoulder:b10,"
        "dpup:b11,dpdown:b12,dpleft:b13,dpright:b14,"
        "leftx:a0,lefty:a1,rightx:a2,righty:a3,lefttrigger:a4,righttrigger:a5,"
    )
    raw.quit()
    assert sdl.SDL_GameControllerAddMapping(mapping.encode()) >= 0

    class VirtualGamepad:
        pg = pygame

        def __init__(self):
            self.index = index
            self.joystick = Joystick(index, deadman=None)
            self.joystick.poll(BOUNDS)
            self.attached = True

        def button(self, button, pressed):
            assert sdl.SDL_JoystickSetVirtualButton(pointer, button, pressed) == 0
            pygame.event.pump()

        def axis(self, axis, value):
            assert sdl.SDL_JoystickSetVirtualAxis(pointer, axis, value) == 0
            pygame.event.pump()

        def poll(self, controls):
            velocity, enabled, keys = self.joystick.poll(controls.bounds)
            controls.joystick(velocity, enabled)
            for key in keys:
                controls.key(key, joystick=True)
            return keys

        def detach(self):
            assert sdl.SDL_JoystickDetachVirtual(index) == 0
            self.attached = False

    pad = VirtualGamepad()
    try:
        yield pad
    finally:
        if pad.attached:
            pad.detach()
        sdl.SDL_JoystickClose(pointer)
        pad.joystick.close()
        pygame.quit()


def test_xbox_mapping_and_exactly_once_pause(gamepad):
    pad = gamepad
    controls = Controls(BOUNDS)
    pad.axis(1, -32768)
    pad.button(4, True)
    assert pad.poll(controls) == []
    np.testing.assert_array_equal(controls.target, 0)
    assert pad.joystick.status(controls) == "HOLD LB"
    pad.button(9, True)
    pad.poll(controls)
    np.testing.assert_allclose(controls.target, [0.8, 0, 0])
    assert pad.joystick.status(controls) == "ACTIVE"
    pad.button(7, True)
    assert pad.poll(controls) == []
    assert not controls.paused
    pad.button(6, True)
    assert pad.poll(controls) == ["p"]
    assert controls.paused
    for _ in range(5):
        pad.pg.event.post(pad.pg.event.Event(
            pad.pg.CONTROLLERBUTTONDOWN, instance_id=pad.joystick.instance, button=pad.pg.CONTROLLER_BUTTON_START,
        ))
        pad.pg.event.post(pad.pg.event.Event(
            pad.pg.JOYBUTTONDOWN, instance_id=pad.joystick.instance, button=6,
        ))
        assert pad.poll(controls) == []
        assert controls.paused
    pad.button(6, False)
    assert pad.poll(controls) == []
    assert controls.paused
    pad.button(6, True)
    assert pad.poll(controls) == ["p"]
    assert not controls.paused
    pad.button(6, False)
    pad.poll(controls)
    pad.button(6, True)
    pad.button(6, False)
    assert pad.poll(controls) == ["p"]
    assert controls.paused


def test_stop_release_and_disconnect_do_not_pause_physics(gamepad):
    torch.set_num_threads(1)
    try:
        checkpoint = find_checkpoint(Path(__file__).resolve().parents[1] / "logs", "mini_cheetah_cpg")
    except FileNotFoundError:
        pytest.skip("Local policy checkpoint is not distributed")
    runtime = Runtime(checkpoint)
    controls = Controls(runtime.bounds)
    pad = gamepad

    def tick():
        pad.poll(controls)
        return advance_simulation(runtime, controls, 1.0)

    try:
        pad.axis(1, -16384)
        pad.button(9, True)
        for _ in range(150):
            assert tick()
        assert runtime.command[0] > 0.3 and not runtime.failed
        pad.button(1, True)
        assert tick()
        np.testing.assert_array_equal(runtime.command, 0)
        assert not controls.paused
        pad.button(1, False)
        pad.button(9, False)
        assert tick()
        pad.button(9, True)
        for _ in range(40):
            assert tick()
        assert runtime.command[0] > 0.3
        pad.button(6, True)
        assert not tick()
        pose = runtime.env.sim.qpos.copy()
        for _ in range(20):
            assert not tick()
        np.testing.assert_array_equal(runtime.env.sim.qpos, pose)
        pad.button(6, False)
        assert not tick()
        pad.button(6, True)
        assert tick()
        pad.button(6, False)
        pad.button(0, True)
        assert tick()
        assert runtime.env.episode_length_buf[0] == 1
        for _ in range(5):
            assert tick()
        assert runtime.env.episode_length_buf[0] == 6
        assert runtime.command[0] > 0
        pad.detach()
        assert tick()
        assert pad.joystick.status(controls) == "DISCONNECTED"
        np.testing.assert_array_equal(runtime.command, 0)
        assert not controls.paused and not runtime.failed
    finally:
        runtime.close()


def test_raw_override_and_binding_conflicts(gamepad):
    gamepad.joystick.close()
    with pytest.raises(ValueError, match="different buttons"):
        Joystick(gamepad.index, deadman=6)
    joystick = Joystick(gamepad.index, deadman=10, pause_button=4)
    gamepad.joystick = joystick
    controls = Controls(BOUNDS)
    joystick.poll(BOUNDS)
    gamepad.button(9, True)
    gamepad.poll(controls)
    assert not joystick.enabled
    gamepad.button(10, True)
    gamepad.poll(controls)
    assert joystick.enabled
    gamepad.button(6, True)
    assert gamepad.poll(controls) == []
    gamepad.button(4, True)
    assert gamepad.poll(controls) == ["p"]
    assert controls.paused


@pytest.mark.parametrize("hold_to_move", [False, True])
def test_reset_and_resume_accept_a_continuously_held_stick(gamepad, hold_to_move):
    torch.set_num_threads(1)
    try:
        checkpoint = find_checkpoint(Path(__file__).resolve().parents[1] / "logs", "mini_cheetah_cpg")
    except FileNotFoundError:
        pytest.skip("Local policy checkpoint is not distributed")
    runtime = Runtime(checkpoint)
    controls = Controls(runtime.bounds)
    pad = gamepad
    if not hold_to_move:
        pad.joystick.close()
        pad.joystick = Joystick(pad.index)

    def tick():
        pad.poll(controls)
        return advance_simulation(runtime, controls, 1.0)

    def assert_walking():
        velocities = []
        for _ in range(200):
            assert tick()
            assert not runtime.failed
            velocities.append(runtime.env.get_tracking_velocity()[0, 0])
        assert runtime.command[0] > 0.3
        assert np.mean(velocities[-100:]) > 0.15
        assert not controls.joystick_inhibited

    try:
        pad.axis(1, -16384)
        if hold_to_move:
            pad.button(9, True)
        assert_walking()
        pad.button(0, True)
        assert tick()
        assert runtime.env.episode_length_buf[0] == 1
        pad.button(0, False)
        assert_walking()
        pad.button(6, True)
        assert not tick()
        pose = runtime.env.sim.qpos.copy()
        for _ in range(20):
            assert not tick()
        np.testing.assert_array_equal(runtime.env.sim.qpos, pose)
        pad.button(6, False)
        assert not tick()
        pad.button(6, True)
        assert tick()
        pad.button(6, False)
        assert_walking()
    finally:
        runtime.close()


def test_held_button_on_connect_and_foreign_events_are_ignored(gamepad):
    pad = gamepad
    controls = Controls(BOUNDS)
    pad.button(6, True)
    pad.joystick.close()
    pad.joystick = Joystick(pad.index, deadman=None)
    assert pad.poll(controls) == []
    assert not controls.paused
    pad.button(6, False)
    assert pad.poll(controls) == []
    pad.pg.event.post(pad.pg.event.Event(
        pad.pg.CONTROLLERBUTTONDOWN, instance_id=pad.joystick.instance + 1000,
        button=pad.pg.CONTROLLER_BUTTON_START,
    ))
    assert pad.poll(controls) == []
    pad.button(6, True)
    assert pad.poll(controls) == ["p"]
    assert controls.paused


def test_controller_cli_defaults_and_invalid_buttons():
    args = parse_args(["--joystick"])
    assert args.axes is None and args.deadman_button is None and args.pause_button is None
    assert not args.hold_to_move
    assert parse_args(["--joystick", "--hold-to-move"]).hold_to_move
    with pytest.raises(SystemExit):
        parse_args(["--pause-button", "-2"])
