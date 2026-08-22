"""Tests for signal conditioning.

The aliasing test is the important one: it pins down the failure mode that would
otherwise silently fabricate fall signal out of SisFall's 200 Hz data.
"""

import numpy as np
import pytest

from shahoshi.signal import adc_to_units, denoise, gravity_split, resample, svm


def sine(f, fs, seconds=8.0, amp=1.0, phase=0.0):
    t = np.arange(int(fs * seconds)) / fs
    return amp * np.sin(2 * np.pi * f * t + phase)


def power_at(sig, f, fs=50.0):
    """Magnitude of the DFT bin nearest `f`, normalized by length."""
    freqs = np.fft.rfftfreq(len(sig), 1 / fs)
    spec = np.abs(np.fft.rfft(sig))
    return spec[np.abs(freqs - f).argmin()] / len(sig)


class TestAdcToUnits:
    def test_adxl345_one_g(self):
        # ADXL345 at +/-16 g, 13-bit: LSB = 32/8192 = 3.90625 mg, so 256 counts = 1 g.
        assert adc_to_units(np.array([256]), 16, 13)[0] == pytest.approx(1.0)

    def test_itg3200_scale(self):
        # ITG3200 at +/-2000 deg/s, 16-bit: LSB = 4000/65536 deg/s.
        assert adc_to_units(np.array([1]), 2000, 16)[0] == pytest.approx(4000 / 65536)

    def test_mma8451q_scale(self):
        # MMA8451Q at +/-8 g, 14-bit: LSB = 16/16384 g, so 1024 counts = 1 g.
        assert adc_to_units(np.array([1024]), 8, 14)[0] == pytest.approx(1.0)

    def test_sign_and_shape_preserved(self):
        raw = np.array([[-256, 0, 256], [512, -512, 0]])
        out = adc_to_units(raw, 16, 13)
        assert out.shape == raw.shape
        assert out[0, 0] == pytest.approx(-1.0)
        assert out[1, 0] == pytest.approx(2.0)

    def test_rejects_bad_bits(self):
        with pytest.raises(ValueError):
            adc_to_units(np.array([1]), 16, 0)


class TestResample:
    def test_preserves_in_band_tone(self):
        """A 5 Hz tone must survive 200 -> 50 Hz with amplitude and frequency intact."""
        y = resample(sine(5.0, 200, seconds=8), 200, 50)
        assert len(y) == pytest.approx(400, abs=2)  # 8 s at 50 Hz
        mid = y[50:-50]  # drop filter edge transients
        assert np.abs(mid).max() == pytest.approx(1.0, abs=0.05)
        freqs = np.fft.rfftfreq(len(mid), 1 / 50)
        assert freqs[np.abs(np.fft.rfft(mid)).argmax()] == pytest.approx(5.0, abs=0.5)

    def test_rejects_out_of_band_tone_instead_of_aliasing_it(self):
        """This is the whole point of the module.

        A 60 Hz tone at 200 Hz is above the 25 Hz Nyquist of the 50 Hz target.
        Plain subsampling folds it to |60 - 50| = 10 Hz, inventing a strong
        in-band signal. Proper resampling must attenuate it instead.
        """
        x = sine(60.0, 200, seconds=8)

        naive = x[::4]                  # the wrong way
        proper = resample(x, 200, 50)   # the right way

        # Naive subsampling produces a large spurious 10 Hz component ...
        assert power_at(naive, 10.0) > 0.2
        # ... and proper resampling leaves essentially nothing there.
        assert power_at(proper, 10.0) < 0.01

        # The out-of-band tone is attenuated by ~60 dB in the interior. The
        # first few samples carry the polyphase FIR's startup transient, so
        # assert on the interior -- and separately pin down that the transient
        # really is confined to the edges, since window extraction later on
        # must be able to trust the interior of a resampled trial.
        assert np.abs(proper[10:-10]).max() < 0.01
        assert np.abs(proper).max() < 0.15

    def test_identity_when_rates_match(self):
        x = sine(3.0, 50)
        assert np.allclose(resample(x, 50, 50), x)

    def test_multichannel_along_axis_0(self):
        x = np.stack([sine(f, 200, seconds=4) for f in (2, 4, 6)], axis=1)
        y = resample(x, 200, 50, axis=0)
        assert y.shape[1] == 3
        assert y.shape[0] == pytest.approx(200, abs=2)

    def test_channels_stay_independent(self):
        """Resampling must not mix channels: a silent channel stays silent."""
        x = np.zeros((800, 3))
        x[:, 0] = sine(5.0, 200, seconds=4)
        y = resample(x, 200, 50, axis=0)
        assert np.abs(y[:, 1]).max() < 1e-9
        assert np.abs(y[:, 2]).max() < 1e-9

    def test_rejects_bad_rates(self):
        with pytest.raises(ValueError):
            resample(sine(1, 50), 0, 50)


class TestGravitySplit:
    def test_static_dc_is_all_gravity(self):
        """A device at rest: 1 g DC on one axis, nothing on the others."""
        acc = np.zeros((500, 3))
        acc[:, 2] = 1.0
        body, gravity = gravity_split(acc, fs=50)
        assert np.abs(gravity[100:-100, 2] - 1.0).max() < 0.01
        assert np.abs(body[100:-100]).max() < 0.01

    def test_dynamic_motion_is_all_body(self):
        """A 2 Hz oscillation is far above the 0.3 Hz cutoff -> body, not gravity."""
        acc = np.zeros((500, 3))
        acc[:, 0] = sine(2.0, 50, seconds=10)
        body, gravity = gravity_split(acc, fs=50)
        assert np.abs(body[100:-100, 0]).max() == pytest.approx(1.0, abs=0.05)
        assert np.abs(gravity[100:-100, 0]).max() < 0.05

    def test_body_plus_gravity_reconstructs_input(self):
        rng = np.random.default_rng(0)
        acc = rng.normal(0, 1, (400, 3)) + np.array([0.0, 0.0, 1.0])
        body, gravity = gravity_split(acc, fs=50)
        assert np.allclose(body + gravity, acc)

    def test_separates_static_offset_from_superposed_motion(self):
        acc = np.zeros((500, 3))
        acc[:, 2] = 1.0 + sine(3.0, 50, seconds=10, amp=0.5)
        body, gravity = gravity_split(acc, fs=50)
        assert np.abs(gravity[100:-100, 2] - 1.0).max() < 0.02
        assert np.abs(body[100:-100, 2]).max() == pytest.approx(0.5, abs=0.03)

    def test_rejects_rate_too_low_for_cutoff(self):
        with pytest.raises(ValueError):
            gravity_split(np.zeros((100, 3)), fs=0.5)


class TestDenoise:
    def test_reduces_high_frequency_noise(self):
        clean = sine(2.0, 50, seconds=10)
        rng = np.random.default_rng(1)
        noisy = clean + rng.normal(0, 0.3, clean.shape)
        out = denoise(noisy, fs=50)
        assert np.abs(out[50:-50] - clean[50:-50]).std() < np.abs(noisy - clean).std()

    def test_preserves_shape_multichannel(self):
        x = np.stack([sine(f, 50, seconds=4) for f in (1, 2, 3)], axis=1)
        assert denoise(x, fs=50).shape == x.shape

    def test_removes_single_sample_spike(self):
        x = np.zeros(200)
        x[100] = 50.0
        assert np.abs(denoise(x, fs=50)).max() < 5.0

    def test_short_signal_does_not_raise(self):
        assert denoise(np.zeros((4, 3)), fs=50).shape == (4, 3)


class TestSvm:
    def test_unit_vectors(self):
        acc = np.array([[1.0, 0, 0], [0, 3.0, 4.0]])
        assert np.allclose(svm(acc), [1.0, 5.0])

    def test_impact_is_the_argmax(self):
        acc = np.zeros((300, 3))
        acc[:, 2] = 1.0                  # resting at 1 g
        acc[177, :] = [5.0, 5.0, 5.0]    # impact
        assert svm(acc).argmax() == 177

    def test_is_rotation_invariant(self):
        """SVM must not depend on device orientation -- that is why we use it
        to find impacts across corpora with different mounting."""
        rng = np.random.default_rng(2)
        acc = rng.normal(0, 1, (50, 3))
        q, _ = np.linalg.qr(rng.normal(0, 1, (3, 3)))
        assert np.allclose(svm(acc), svm(acc @ q.T))
