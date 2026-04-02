"""Tests for scheduling.proposed module (stdlib-only)."""

from pathlib import Path
import sys
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class SchedulingProposedTest(unittest.TestCase):
    def test_import(self):
        from scheduling.proposed import ProposedScheduler

        self.assertTrue(callable(ProposedScheduler))

    def test_empty_candidates(self):
        from scheduling.proposed import ProposedScheduler

        scheduler = ProposedScheduler()
        selected, state = scheduler.select_next([])
        self.assertIsNone(selected)
        self.assertEqual(state, {"index": 0})

    def test_order_target_ids_sorts_by_pan_descending(self):
        from scheduling.proposed import ProposedScheduler

        scheduler = ProposedScheduler()
        targets = {
            3: (12.0, 0.0),
            1: (-3.0, 1.0),
            4: (12.0, 2.0),
            2: (5.0, -1.0),
        }

        ordered = scheduler.order_target_ids(targets)

        self.assertEqual(ordered, [3, 4, 2, 1])

    def test_led_state_to_battery_coeff_mapping(self):
        from scheduling.proposed import led_state_to_battery_coeff

        self.assertEqual(led_state_to_battery_coeff("R"), 0.75)
        self.assertEqual(led_state_to_battery_coeff("B"), 0.50)
        self.assertEqual(led_state_to_battery_coeff("G"), 0.25)

    def test_normalize_target_coeffs_uses_max(self):
        from scheduling.proposed import normalize_target_coeffs

        normalized = normalize_target_coeffs({1: 2.0, 2: 4.0, 3: 1.0}, ordered_track_ids=[1, 2, 3])

        self.assertEqual(normalized, {1: 0.5, 2: 1.0, 3: 0.25})

    def test_compute_frame_allocations_uses_score_ratio(self):
        from scheduling.proposed import compute_frame_allocations

        allocations, scores = compute_frame_allocations(
            total_frame_time=40.0,
            ordered_track_ids=[1, 2],
            fixed_target_coeffs={1: 1.0, 2: 0.5},
            battery_coeff_prev={1: 0.5, 2: 0.75},
        )

        self.assertAlmostEqual(scores[1], 0.5)
        self.assertAlmostEqual(scores[2], 0.375)
        self.assertAlmostEqual(allocations[1], 22.8571428571, places=6)
        self.assertAlmostEqual(allocations[2], 17.1428571429, places=6)
        self.assertAlmostEqual(sum(allocations.values()), 40.0, places=6)

    def test_invalid_green_to_red_transition_is_blocked(self):
        from scheduling.proposed import sample_or_update_battery_state_for_target

        update = sample_or_update_battery_state_for_target(
            track_id=3,
            previous_state="G",
            sampled_state="R",
        )

        self.assertEqual(update["next_state"], "G")
        self.assertEqual(update["next_coeff"], 0.25)
        self.assertFalse(update["valid"])

    def test_initialize_fixed_target_coeffs_from_csv_groups_and_normalizes(self):
        from scheduling.proposed import initialize_fixed_target_coeffs_from_csv

        csv_path = PROJECT_ROOT / ".tmp_test_scheduling_proposed.csv"
        try:
            csv_path.write_text(
                "\n".join(
                    [
                        "track_id,mean",
                        "10,5.0",
                        "10,7.0",
                        "20,2.0",
                    ]
                ),
                encoding="utf-8",
            )

            result = initialize_fixed_target_coeffs_from_csv(
                csv_path=str(csv_path),
                track_id_members={1: (10,), 2: (20,)},
                ordered_track_ids=[1, 2],
            )

            self.assertEqual(result["mean_field"], "mean")
            self.assertAlmostEqual(result["raw"][1], 6.0)
            self.assertAlmostEqual(result["raw"][2], 2.0)
            self.assertAlmostEqual(result["normalized"][1], 1.0)
            self.assertAlmostEqual(result["normalized"][2], 2.0 / 6.0)
        finally:
            if csv_path.exists():
                csv_path.unlink()


if __name__ == "__main__":
    unittest.main()
