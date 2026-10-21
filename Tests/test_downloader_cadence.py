"""Fixed scientific cases for the satellite export cadence policy."""

import unittest

import numpy as np
import pandas as pd

from downloader_cadence import average_subsecond, merge_products, unique_epoch_rows


class CadenceChecks(unittest.TestCase):
    def test_identical_hourly_boundary_rows_preserve_native_data_and_missingness(self):
        times = pd.to_datetime(["2025-06-01T00:00Z", "2025-06-01T01:00Z"])
        original = pd.DataFrame({"V": [691.130005, 700.], "N_p": [2.3259, np.nan],
                                 "V_Z_GSE": [np.nan, np.nan],
                                 "source_dataset": ["AC_K1_SWE", "AC_K1_SWE"]}, index=times)
        original.attrs = {"frame": "GSE", "units": "km/s"}
        repeated = pd.concat([original.iloc[:1], original])
        result = unique_epoch_rows(repeated, "ACE plasma")
        pd.testing.assert_frame_equal(result, original, check_exact=True)
        self.assertEqual(result.attrs, original.attrs)
        pd.testing.assert_frame_equal(average_subsecond(result), original, check_exact=True)

    def test_conflicting_same_epoch_values_and_missingness_are_not_discarded(self):
        for values in [[691., 692.], [691., np.nan]]:
            frame = pd.DataFrame({"V": values}, index=pd.to_datetime(["2025-06-01T00:00Z"]*2))
            with self.assertRaisesRegex(ValueError, "ACE plasma: conflicting values"):
                unique_epoch_rows(frame, "ACE plasma")

    def test_repeated_subsecond_sample_does_not_bias_the_one_second_mean(self):
        source = pd.DataFrame({"B_X_GSE": [3., -3., 4., 8.], "B": [3., 3., 4., 8.]},
                              index=pd.to_datetime(["2025-06-01T00:00:00.000Z", "2025-06-01T00:00:00.128Z",
                                                    "2025-06-01T00:00:00.256Z", "2025-06-01T00:00:04.000Z"]))
        repeated = pd.concat([source.iloc[:1], source])
        actual = average_subsecond(unique_epoch_rows(repeated, "MAG"))
        expected = average_subsecond(source)
        pd.testing.assert_frame_equal(actual, expected, check_exact=True)

    def test_vector_means_magnitude_missing_samples_and_gaps(self):
        times = pd.to_datetime([
            "2026-01-19T22:00:00.000Z", "2026-01-19T22:00:00.128Z",
            "2026-01-19T22:00:00.256Z", "2026-01-19T22:00:04.000Z",
            "2026-01-19T22:00:04.128Z", "2026-01-19T22:00:05.000Z",
        ])
        source = pd.DataFrame({
            "B_X_GSE": [3., -3., 0., 4., np.nan, np.nan],
            "B_Y_GSE": [0., 0., 0., 0., np.nan, np.nan],
            "B_Z_GSE": [0., 0., 0., 3., np.nan, np.nan],
            "B": [3., 3., 0., 5., np.nan, np.nan],
            "B_X_GSE_error": [.3, .3, .3, .3, np.nan, np.nan],
        }, index=pd.DatetimeIndex(times, name="Epoch"))
        source.attrs = {"units": "nT", "frame": "GSE"}
        result = average_subsecond(source)
        expected = pd.DataFrame({
            "B_X_GSE": [0., 4., np.nan], "B_Y_GSE": [0., 0., np.nan],
            "B_Z_GSE": [0., 3., np.nan], "B": [2., 5., np.nan],
            "B_X_GSE_error": [.3, .3, np.nan],
        }, index=pd.DatetimeIndex(times[[0, 3, 5]], name="Epoch"))
        pd.testing.assert_frame_equal(result, expected, atol=1e-14, rtol=0)
        self.assertEqual(result.attrs, source.attrs)

    def test_slower_products_keep_exact_values_and_fractional_timestamps(self):
        for seconds in [1, 5, 10, 60]:
            times = pd.date_range("2026-01-19T22:00:00.250Z", periods=4,
                                  freq=pd.Timedelta(seconds=seconds), name="Epoch")
            source = pd.DataFrame({"N_p": [2., np.nan, 4., 6.]}, index=times)
            pd.testing.assert_frame_equal(average_subsecond(source), source)

    def test_independent_one_second_products_are_not_averaged_after_join(self):
        times = pd.date_range("2026-01-19T22:00:00Z", periods=3, freq="1s")
        magnetic = pd.DataFrame({"B": [1., 2., 3.]}, index=times)
        plasma = pd.DataFrame({"V": [300., 400., 500.]},
                              index=times + pd.Timedelta(milliseconds=500))
        merged = merge_products({"magnetic": magnetic, "plasma": plasma},
                                average=("magnetic", "plasma"))
        self.assertEqual(len(merged), 6)
        pd.testing.assert_series_equal(merged["V"].dropna(), plasma["V"],
                                       check_freq=False, check_names=False)
        self.assertTrue(merged.loc[magnetic.index, "V"].isna().all())

    def test_merge_keeps_positions_native_and_deduplicates_before_science_means(self):
        magnetic = pd.DataFrame(
            {"B_X_GSE": [3., -3., 0., 0., 4., np.nan], "B": [3., 3., 0., 0., 4., np.nan]},
            index=pd.to_datetime(["2025-06-01T00:00:00.100Z", "2025-06-01T00:00:00.200Z",
                                  "2025-06-01T00:00:00.300Z", "2025-06-01T00:00:00.400Z",
                                  "2025-06-01T00:00:04.100Z", "2025-06-01T00:00:06.100Z"]))
        repeated = pd.concat([magnetic.iloc[:1], magnetic])
        plasma = pd.DataFrame({"V": [700., 710.]},
                              index=pd.to_datetime(["2025-06-01T00:00:00.500Z", "2025-06-01T00:00:05.500Z"]))
        position = pd.DataFrame({"X_GSE": [200., 201.]},
                                index=pd.to_datetime(["2025-06-01T00:00:00.000Z", "2025-06-01T00:00:30.250Z"]))
        original = repeated.copy(deep=True)
        actual = merge_products({"mag": repeated, "plasma": plasma, "position": position},
                                average=("mag", "plasma"))
        expected = pd.DataFrame(
            {"B_X_GSE": [0., np.nan, 4., np.nan, np.nan],
             "B": [1.5, np.nan, 4., np.nan, np.nan],
             "V": [np.nan, 700., np.nan, 710., np.nan],
             "X_GSE": [200., np.nan, np.nan, np.nan, 201.]},
            index=pd.DatetimeIndex(pd.to_datetime(["2025-06-01T00:00:00.000Z", "2025-06-01T00:00:00.500Z",
                                                  "2025-06-01T00:00:04.000Z", "2025-06-01T00:00:05.500Z",
                                                  "2025-06-01T00:00:30.250Z"]), name="Epoch"))
        pd.testing.assert_frame_equal(actual, expected, check_exact=True)
        pd.testing.assert_frame_equal(repeated, original, check_exact=True)

    def test_merge_retains_native_quality_flags_and_requested_empty_rows(self):
        times = pd.date_range("2025-06-01T05:00:00Z", periods=3, freq="10s", name="time")
        magnetic = pd.DataFrame({"B": [3., np.nan, np.nan], "quality": [1., 0., np.nan]}, index=times)
        actual = merge_products({"L2 magnetic": magnetic}, index_name="time", drop_empty=False)
        pd.testing.assert_frame_equal(actual, magnetic, check_exact=True)
        conflicting = pd.concat([magnetic.iloc[:1], magnetic])
        conflicting.iloc[0, 0] = 4.
        with self.assertRaisesRegex(ValueError, "L2 magnetic: conflicting values"):
            merge_products({"L2 magnetic": conflicting})

    def test_datetime_storage_unit_does_not_change_cadence(self):
        times = pd.date_range("2026-01-19T22:00:00Z", periods=8, freq="128ms")
        source = pd.DataFrame({"V": np.arange(8.)}, index=times)
        microseconds = source.copy()
        microseconds.index = microseconds.index.as_unit("us")
        pd.testing.assert_frame_equal(average_subsecond(microseconds),
                                      average_subsecond(source), check_index_type=False)


if __name__ == "__main__":
    unittest.main()
