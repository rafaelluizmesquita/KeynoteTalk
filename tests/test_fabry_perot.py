"""Independent Fabry--Pérot forward-model and retrieval checks."""
import numpy as np
from sbgea_keynote.instruments.fabry_perot import (simulate_interferogram,retrieve_line_of_sight_wind,
                         expected_interferogram)


def test_synthetic_interferogram_is_reproducible_and_wind_sensitive():
    a=simulate_interferogram(100.,seed=2)
    b=simulate_interferogram(100.,seed=2)
    assert np.array_equal(a["counts"],b["counts"])
    assert np.allclose(a["variance_counts2"],np.maximum(a["counts"],0)+3.0**2)
    assert np.linalg.norm(expected_interferogram(100.)-expected_interferogram(0.))>1


def test_fabry_perot_recovers_known_doppler_wind():
    obs=simulate_interferogram(120.,seed=2026)
    fit=retrieve_line_of_sight_wind(obs)
    assert fit["success"]
    assert abs(fit["wind_los_m_s"]-120.)<8.
    assert 0<fit["wind_std_m_s"]<5.
