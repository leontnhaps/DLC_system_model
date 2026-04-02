"""Proposed scheduling algorithm helpers and state."""

from __future__ import annotations

import csv
from collections import defaultdict

from scheduling.base import SchedulingAlgorithm


BATTERY_COEFF_MAP = {
    "R": 0.75,
    "B": 0.50,
    "G": 0.25,
}
DEFAULT_BATTERY_STATE = "R"
DEFAULT_MEAN_FIELD_CANDIDATES = (
    "mean",
    "mean_delta",
    "laser_mean_delta",
    "final_phase3_response_mean",
)


def _canonical_led_state(value, default=None):
    """Normalize LED labels to one of ``R/G/B``."""
    text = str(value or "").strip().upper()
    if text in ("R", "RED"):
        return "R"
    if text in ("G", "GREEN"):
        return "G"
    if text in ("B", "BLUE"):
        return "B"
    return default


def normalize_target_coeffs(raw_coeffs, ordered_track_ids=None, default_value=1.0):
    """Normalize raw target coefficients by their maximum value."""
    ordered_ids = list(ordered_track_ids or raw_coeffs.keys())
    cleaned = {}
    max_raw = 0.0

    for track_id in ordered_ids:
        raw_value = raw_coeffs.get(track_id, default_value)
        try:
            raw_value = float(raw_value)
        except Exception:
            raw_value = float(default_value)
        raw_value = max(0.0, raw_value)
        cleaned[track_id] = raw_value
        if raw_value > max_raw:
            max_raw = raw_value

    if max_raw <= 0.0:
        return {track_id: 1.0 for track_id in ordered_ids}
    return {track_id: (cleaned[track_id] / max_raw) for track_id in ordered_ids}


def initialize_fixed_target_coeffs_from_csv(
    csv_path,
    track_id_members=None,
    ordered_track_ids=None,
    mean_field_candidates=None,
):
    """Compute frozen ``C_n`` from the scan CSV mean-like field."""
    track_id_members = {
        int(track_id): tuple(int(v) for v in members)
        for track_id, members in dict(track_id_members or {}).items()
    }
    ordered_ids = list(ordered_track_ids or track_id_members.keys())
    mean_field_candidates = tuple(mean_field_candidates or DEFAULT_MEAN_FIELD_CANDIDATES)
    raw_by_csv_track = defaultdict(list)
    selected_mean_field = None

    if csv_path:
        try:
            with open(csv_path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                fieldnames = tuple(reader.fieldnames or ())
                for field_name in mean_field_candidates:
                    if field_name in fieldnames:
                        selected_mean_field = field_name
                        break

                if selected_mean_field is not None:
                    for row in reader:
                        try:
                            csv_track_id = int(float(row.get("track_id", "") or 0))
                        except Exception:
                            continue
                        raw_mean = row.get(selected_mean_field)
                        if raw_mean in ("", None):
                            continue
                        try:
                            raw_by_csv_track[csv_track_id].append(float(raw_mean))
                        except Exception:
                            continue
        except Exception:
            selected_mean_field = None

    raw_coeffs = {}
    for track_id in ordered_ids:
        member_ids = tuple(track_id_members.get(track_id, (track_id,)))
        member_raws = []
        for member_id in member_ids:
            values = raw_by_csv_track.get(int(member_id), ())
            if values:
                member_raws.append(sum(values) / float(len(values)))
        if member_raws:
            raw_coeffs[int(track_id)] = sum(member_raws) / float(len(member_raws))
        else:
            raw_coeffs[int(track_id)] = 1.0

    normalized = normalize_target_coeffs(raw_coeffs, ordered_track_ids=ordered_ids, default_value=1.0)
    return {
        "raw": raw_coeffs,
        "normalized": normalized,
        "mean_field": selected_mean_field,
    }


def led_state_to_battery_coeff(led_state, default_state=DEFAULT_BATTERY_STATE):
    """Convert LED state to the required battery urgency coefficient."""
    canonical = _canonical_led_state(led_state, default=default_state)
    return float(BATTERY_COEFF_MAP.get(canonical, BATTERY_COEFF_MAP[DEFAULT_BATTERY_STATE]))


def compute_frame_allocations(total_frame_time, ordered_track_ids, fixed_target_coeffs, battery_coeff_prev):
    """Compute ``Score_n^k`` and ``t_n^k`` for one frame."""
    ordered_ids = [int(track_id) for track_id in ordered_track_ids or []]
    try:
        total_frame_time = max(0.0, float(total_frame_time))
    except Exception:
        total_frame_time = 0.0

    scores = {}
    for track_id in ordered_ids:
        coeff_c = max(0.0, float((fixed_target_coeffs or {}).get(track_id, 1.0)))
        coeff_b = max(0.0, float((battery_coeff_prev or {}).get(track_id, BATTERY_COEFF_MAP[DEFAULT_BATTERY_STATE])))
        scores[track_id] = coeff_b * coeff_c

    score_sum = sum(scores.values())
    allocations = {}
    if not ordered_ids:
        return allocations, scores

    if score_sum <= 0.0:
        equal_alloc = total_frame_time / float(len(ordered_ids)) if ordered_ids else 0.0
        return {track_id: equal_alloc for track_id in ordered_ids}, scores

    allocated_sum = 0.0
    last_track_id = ordered_ids[-1]
    for track_id in ordered_ids:
        if track_id == last_track_id:
            allocations[track_id] = max(0.0, total_frame_time - allocated_sum)
            continue
        alloc = total_frame_time * (scores[track_id] / score_sum)
        alloc = max(0.0, float(alloc))
        allocations[track_id] = alloc
        allocated_sum += alloc
    return allocations, scores


def sample_or_update_battery_state_for_target(track_id, previous_state, sampled_state):
    """Validate sampled LED state and convert it into ``B_n``."""
    prev_canonical = _canonical_led_state(previous_state, default=DEFAULT_BATTERY_STATE)
    sampled_canonical = _canonical_led_state(sampled_state, default=None)
    valid = True
    reason = "sampled"

    if sampled_canonical is None:
        next_state = prev_canonical
        valid = False
        reason = "missing_or_invalid"
    elif prev_canonical == "G" and sampled_canonical == "R":
        next_state = prev_canonical
        valid = False
        reason = "blocked_green_to_red"
    else:
        next_state = sampled_canonical

    return {
        "track_id": int(track_id),
        "previous_state": prev_canonical,
        "sampled_state": sampled_canonical,
        "next_state": next_state,
        "next_coeff": led_state_to_battery_coeff(next_state),
        "valid": bool(valid),
        "reason": reason,
    }


def finalize_frame_and_prepare_next(state):
    """Finalize frame ``k`` and roll ``B_n^k`` into the next frame as previous state."""
    execution_order = [int(track_id) for track_id in state.get("execution_order", [])]
    battery_state_prev = dict(state.get("battery_state_prev", {}) or {})
    battery_coeff_prev = dict(state.get("battery_coeff_prev", {}) or {})
    battery_state_next = dict(state.get("battery_state_next", {}) or {})
    battery_coeff_next = dict(state.get("battery_coeff_next", {}) or {})

    for track_id in execution_order:
        if track_id not in battery_state_next:
            battery_state_next[track_id] = battery_state_prev.get(track_id, DEFAULT_BATTERY_STATE)
        if track_id not in battery_coeff_next:
            battery_coeff_next[track_id] = battery_coeff_prev.get(
                track_id,
                led_state_to_battery_coeff(battery_state_next[track_id]),
            )

    state["battery_state_prev"] = dict(battery_state_next)
    state["battery_coeff_prev"] = dict(battery_coeff_next)
    state["battery_state_next"] = {}
    state["battery_coeff_next"] = {}
    state["frame_index"] = int(state.get("frame_index", 1)) + 1
    return state


class ProposedScheduler(SchedulingAlgorithm):
    """Proposed scheduler that preserves RR execution order and changes only time allocation."""

    def select_next(self, candidates, state=None):
        items = list(candidates or [])
        if not items:
            return None, {"index": 0}

        index = 0
        if isinstance(state, dict):
            try:
                index = int(state.get("index", 0))
            except Exception:
                index = 0

        index %= len(items)
        selected = items[index]
        next_state = {"index": (index + 1) % len(items)}
        return selected, next_state

    def order_target_ids(self, targets):
        items = list((targets or {}).items())
        items.sort(key=lambda item: (-float(item[1][0]), int(item[0])))
        return [track_id for track_id, _target in items]

    def initialize_state(
        self,
        total_frame_time,
        ordered_track_ids,
        csv_path=None,
        track_id_members=None,
        initial_led_states=None,
    ):
        ordered_ids = [int(track_id) for track_id in ordered_track_ids or []]
        coeff_bundle = initialize_fixed_target_coeffs_from_csv(
            csv_path=csv_path,
            track_id_members=track_id_members,
            ordered_track_ids=ordered_ids,
        )
        battery_state_prev = {}
        battery_coeff_prev = {}
        initial_led_states = dict(initial_led_states or {})
        for track_id in ordered_ids:
            state_value = _canonical_led_state(initial_led_states.get(track_id), default=DEFAULT_BATTERY_STATE)
            battery_state_prev[track_id] = state_value
            battery_coeff_prev[track_id] = led_state_to_battery_coeff(state_value)

        return {
            "frame_index": 1,
            "total_frame_time": float(max(0.0, float(total_frame_time))),
            "execution_order": ordered_ids,
            "fixed_target_coeffs": dict(coeff_bundle["normalized"]),
            "fixed_target_raw_coeffs": dict(coeff_bundle["raw"]),
            "mean_field_name": coeff_bundle.get("mean_field"),
            "battery_state_prev": battery_state_prev,
            "battery_coeff_prev": battery_coeff_prev,
            "battery_state_next": {},
            "battery_coeff_next": {},
            "frame_allocations": {},
            "frame_scores": {},
        }

    def build_frame_plan(self, state, total_frame_time=None, execution_order=None):
        if total_frame_time is not None:
            state["total_frame_time"] = float(max(0.0, float(total_frame_time)))
        if execution_order is not None:
            state["execution_order"] = [int(track_id) for track_id in execution_order]

        allocations, scores = compute_frame_allocations(
            total_frame_time=state.get("total_frame_time", 0.0),
            ordered_track_ids=state.get("execution_order", ()),
            fixed_target_coeffs=state.get("fixed_target_coeffs", {}),
            battery_coeff_prev=state.get("battery_coeff_prev", {}),
        )
        state["frame_allocations"] = dict(allocations)
        state["frame_scores"] = dict(scores)
        state["battery_state_next"] = {}
        state["battery_coeff_next"] = {}
        return {
            "frame_index": int(state.get("frame_index", 1)),
            "execution_order": list(state.get("execution_order", ())),
            "allocations": dict(allocations),
            "scores": dict(scores),
            "fixed_target_coeffs": dict(state.get("fixed_target_coeffs", {})),
            "battery_state_prev": dict(state.get("battery_state_prev", {})),
            "battery_coeff_prev": dict(state.get("battery_coeff_prev", {})),
        }

    @staticmethod
    def get_sampling_elapsed(allocation_s):
        """Return when to sample within the slice, measured from slice start."""
        try:
            allocation_s = max(0.0, float(allocation_s))
        except Exception:
            allocation_s = 0.0
        if allocation_s <= 0.0:
            return None
        lead_time = 10.0 if allocation_s >= 10.0 else (allocation_s * 0.1)
        return max(0.0, allocation_s - lead_time)

    def sample_or_update_battery_state_for_target(self, state, track_id, sampled_state):
        previous_state = dict(state.get("battery_state_prev", {}) or {}).get(track_id, DEFAULT_BATTERY_STATE)
        update = sample_or_update_battery_state_for_target(track_id, previous_state, sampled_state)
        state.setdefault("battery_state_next", {})[int(track_id)] = update["next_state"]
        state.setdefault("battery_coeff_next", {})[int(track_id)] = float(update["next_coeff"])
        return update

    def finalize_frame_and_prepare_next(self, state):
        return finalize_frame_and_prepare_next(state)

    def log_frame_summary(self, state):
        frame_index = int(state.get("frame_index", 1))
        execution_order = list(state.get("execution_order", ()))
        print(
            f"[Scheduling-Proposed] Frame {frame_index} "
            f"order={execution_order} mean_field={state.get('mean_field_name') or 'fallback'}"
        )
        for track_id in execution_order:
            coeff_c = float((state.get("fixed_target_coeffs", {}) or {}).get(track_id, 1.0))
            prev_state = str((state.get("battery_state_prev", {}) or {}).get(track_id, DEFAULT_BATTERY_STATE))
            prev_coeff = float((state.get("battery_coeff_prev", {}) or {}).get(track_id, led_state_to_battery_coeff(prev_state)))
            score = float((state.get("frame_scores", {}) or {}).get(track_id, 0.0))
            alloc = float((state.get("frame_allocations", {}) or {}).get(track_id, 0.0))
            next_state = str((state.get("battery_state_next", {}) or {}).get(track_id, prev_state))
            next_coeff = float((state.get("battery_coeff_next", {}) or {}).get(track_id, led_state_to_battery_coeff(next_state)))
            print(
                "[Scheduling-Proposed] frame={frame} track_id={track} "
                "C_n={coeff_c:.6f} prev_state={prev_state} prev_B={prev_coeff:.2f} "
                "score={score:.6f} alloc={alloc:.3f}s next_state={next_state} next_B={next_coeff:.2f}".format(
                    frame=frame_index,
                    track=int(track_id),
                    coeff_c=coeff_c,
                    prev_state=prev_state,
                    prev_coeff=prev_coeff,
                    score=score,
                    alloc=alloc,
                    next_state=next_state,
                    next_coeff=next_coeff,
                )
            )


__all__ = [
    "BATTERY_COEFF_MAP",
    "DEFAULT_BATTERY_STATE",
    "DEFAULT_MEAN_FIELD_CANDIDATES",
    "ProposedScheduler",
    "compute_frame_allocations",
    "finalize_frame_and_prepare_next",
    "initialize_fixed_target_coeffs_from_csv",
    "led_state_to_battery_coeff",
    "normalize_target_coeffs",
    "sample_or_update_battery_state_for_target",
]
