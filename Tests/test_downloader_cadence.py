"""Fixed scientific cases for the satellite export cadence policy."""

import unittest

import numpy as np
import pandas as pd

from downloader_cadence import average_subsecond


class CadenceChecks(unittest.TestCase):
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
        merged = pd.concat([average_subsecond(magnetic), average_subsecond(plasma)],
                           axis=1).sort_index()
        self.assertEqual(len(merged), 6)
        pd.testing.assert_series_equal(merged["V"].dropna(), plasma["V"], check_freq=False)
        self.assertTrue(merged.loc[magnetic.index, "V"].isna().all())

    def test_datetime_storage_unit_does_not_change_cadence(self):
        times = pd.date_range("2026-01-19T22:00:00Z", periods=8, freq="128ms")
        source = pd.DataFrame({"V": np.arange(8.)}, index=times)
        microseconds = source.copy()
        microseconds.index = microseconds.index.as_unit("us")
        pd.testing.assert_frame_equal(average_subsecond(microseconds),
                                      average_subsecond(source), check_index_type=False)


if __name__ == "__main__":
    unittest.main()
