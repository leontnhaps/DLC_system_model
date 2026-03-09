"""Centralized protocol constants/builders for GUI command payloads."""

from typing import Any


# GUI -> control channel command strings (from app/window.py)
CMD_PREVIEW = "preview"
CMD_SNAP = "snap"
CMD_SCAN_RUN = "scan_run"
CMD_SCAN_STOP = "scan_stop"
CMD_MOVE = "move"
CMD_LED = "led"
CMD_LASER = "laser"
CMD_IR_CUT = "ir_cut"

ALL_COMMANDS = (
    CMD_PREVIEW,
    CMD_SNAP,
    CMD_SCAN_RUN,
    CMD_SCAN_STOP,
    CMD_MOVE,
    CMD_LED,
    CMD_LASER,
    CMD_IR_CUT,
)

# Local GUI event-bus tags (from network/event_handlers surface)
EVT_TAG_TOAST = "toast"
EVT_TAG_EVT = "evt"
EVT_TAG_PREVIEW = "preview"
EVT_TAG_SAVED = "saved"

ALL_EVENT_TAGS = (
    EVT_TAG_TOAST,
    EVT_TAG_EVT,
    EVT_TAG_PREVIEW,
    EVT_TAG_SAVED,
)


def build_preview_cmd(enable, width, height, fps, quality, shutter_speed, analogue_gain):
    return {
        "cmd": CMD_PREVIEW,
        "enable": enable,
        "width": width,
        "height": height,
        "fps": fps,
        "quality": quality,
        "shutter_speed": shutter_speed,
        "analogue_gain": analogue_gain,
    }


def build_snap_cmd(width, height, quality, save, shutter_speed=None, analogue_gain=None):
    cmd = {
        "cmd": CMD_SNAP,
        "width": width,
        "height": height,
        "quality": quality,
        "save": save,
    }
    if shutter_speed is not None:
        cmd["shutter_speed"] = shutter_speed
    if analogue_gain is not None:
        cmd["analogue_gain"] = analogue_gain
    return cmd


def build_scan_run_cmd(session, **params: Any):
    return {
        "cmd": CMD_SCAN_RUN,
        "session": session,
        **params,
    }


def build_scan_stop_cmd():
    return {"cmd": CMD_SCAN_STOP}


def build_move_cmd(pan, tilt, speed, acc):
    return {
        "cmd": CMD_MOVE,
        "pan": pan,
        "tilt": tilt,
        "speed": speed,
        "acc": acc,
    }


def build_led_cmd(value):
    return {
        "cmd": CMD_LED,
        "value": value,
    }


def build_laser_cmd(value):
    return {
        "cmd": CMD_LASER,
        "value": value,
    }


def build_ir_cut_cmd(mode):
    return {
        "cmd": CMD_IR_CUT,
        "mode": mode,
    }


__all__ = [
    "CMD_PREVIEW",
    "CMD_SNAP",
    "CMD_SCAN_RUN",
    "CMD_SCAN_STOP",
    "CMD_MOVE",
    "CMD_LED",
    "CMD_LASER",
    "CMD_IR_CUT",
    "ALL_COMMANDS",
    "EVT_TAG_TOAST",
    "EVT_TAG_EVT",
    "EVT_TAG_PREVIEW",
    "EVT_TAG_SAVED",
    "ALL_EVENT_TAGS",
    "build_preview_cmd",
    "build_snap_cmd",
    "build_scan_run_cmd",
    "build_scan_stop_cmd",
    "build_move_cmd",
    "build_led_cmd",
    "build_laser_cmd",
    "build_ir_cut_cmd",
]
