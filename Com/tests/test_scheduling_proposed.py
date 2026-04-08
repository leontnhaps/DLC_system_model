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

        self.assertEqual(led_state_to_battery_coeff("000"), 1.0)
        self.assertEqual(led_state_to_battery_coeff("001"), 0.875)
        self.assertEqual(led_state_to_battery_coeff("010"), 0.75)
        self.assertEqual(led_state_to_battery_coeff("011"), 0.625)
        self.assertEqual(led_state_to_battery_coeff("100"), 0.50)
        self.assertEqual(led_state_to_battery_coeff("101"), 0.375)
        self.assertEqual(led_state_to_battery_coeff("110"), 0.25)
        self.assertEqual(led_state_to_battery_coeff("111"), 0.125)

    def test_normalize_target_coeffs_uses_max(self):
        from scheduling.proposed import normalize_target_coeffs

        normalized = normalize_target_coeffs({1: 2.0, 2: 4.0, 3: 1.0}, ordered_track_ids=[1, 2, 3])

        self.assertEqual(normalized, {1: 0.5, 2: 1.0, 3: 0.25})

    def test_compute_frame_allocations_uses_score_ratio(self):
        from scheduling.proposed import compute_frame_allocations

        allocations, scores = compute_frame_allocations(
            total_frame_time=40.0,
            ordered_track_ids=[1, 2],
            fixed_target_coeffs={1: 100.0, 2: 0.001},
            battery_coeff_prev={1: 0.5, 2: 0.75},
        )

        self.assertAlmostEqual(scores[1], 0.5)
        self.assertAlmostEqual(scores[2], 0.75)
        self.assertAlmostEqual(allocations[1], 16.0, places=6)
        self.assertAlmostEqual(allocations[2], 24.0, places=6)
        self.assertAlmostEqual(sum(allocations.values()), 40.0, places=6)

    def test_sampling_elapsed_points_repeat_every_10_seconds(self):
        from scheduling.proposed import ProposedScheduler

        self.assertEqual(ProposedScheduler.get_sampling_elapsed_points(74.0, interval_s=10.0), [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0])
        self.assertEqual(ProposedScheduler.get_sampling_elapsed_points(58.0, interval_s=10.0), [10.0, 20.0, 30.0, 40.0, 50.0])
        self.assertEqual(ProposedScheduler.get_sampling_elapsed_points(9.9, interval_s=10.0), [])

    def test_multiple_samples_in_one_frame_always_compare_to_previous_frame(self):
        from scheduling.proposed import ProposedScheduler

        scheduler = ProposedScheduler()
        state = scheduler.initialize_state(
            total_frame_time=240.0,
            ordered_track_ids=[1],
            initial_led_states={1: "111"},
        )

        update1 = scheduler.sample_or_update_battery_state_for_target(state, 1, "110")
        update2 = scheduler.sample_or_update_battery_state_for_target(state, 1, "101")

        self.assertEqual(update1["next_state"], "110")
        self.assertEqual(update2["previous_state"], "111")
        self.assertEqual(update2["next_state"], "111")
        self.assertEqual(state["battery_state_next"][1], "111")

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

    def test_non_adjacent_3bit_transition_is_blocked(self):
        from scheduling.proposed import sample_or_update_battery_state_for_target

        update = sample_or_update_battery_state_for_target(
            track_id=4,
            previous_state="111",
            sampled_state="011",
        )

        self.assertEqual(update["next_state"], "111")
        self.assertEqual(update["reason"], "blocked_non_adjacent_bit_transition")
        self.assertFalse(update["valid"])

    def test_increasing_3bit_transition_is_blocked(self):
        from scheduling.proposed import sample_or_update_battery_state_for_target

        update = sample_or_update_battery_state_for_target(
            track_id=5,
            previous_state="101",
            sampled_state="110",
        )

        self.assertEqual(update["next_state"], "101")
        self.assertEqual(update["reason"], "blocked_increasing_bit_transition")
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
