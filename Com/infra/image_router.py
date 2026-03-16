"""Saved-image routing logic extracted from event handlers."""

import time

from app_config import SAVE_DIR


def route_saved_image(app, name: str, data: bytes) -> None:
    """Route a saved image to pointing/scan/default flows."""
    if hasattr(app, "_aiming_active") and app._aiming_active:
        if name.startswith("pointing_"):
            print(f"[POINTING_IMG] Routing to pointing handler: {name}")
            app._on_pointing_image_received(name, data)
            app._set_preview(data)
            return

    # Scheduling 등 blocking snap 대기자에게도 알림 (소비하지는 않음)
    if hasattr(app, "_notify_blocking_snap_saved"):
        app._notify_blocking_snap_saved(name, data)

    if app.scan_ctrl.is_active():
        saved_path = app.scan_ctrl.save_image(name, data)
        if saved_path:
            print(f"[SCAN_SAVE] {saved_path}")
            # done 이후 finalize idle 기준 갱신
            app._last_scan_image_ts = time.monotonic()
        app._set_preview(data)
        return

    SAVE_DIR.mkdir(exist_ok=True)
    save_path = SAVE_DIR / name
    with open(save_path, "wb") as f:
        f.write(data)
    print(f"[SAVE] {save_path}")

    if hasattr(app, "info_label"):
        app.info_label.config(text=f"💾 저장됨: {name}")

    # 수동 Snap 완료 콜백 (Preview 자동 복구용)
    if name.startswith("snap_") and hasattr(app, "_on_manual_snap_saved"):
        app._on_manual_snap_saved(name)

    app._set_preview(data)
