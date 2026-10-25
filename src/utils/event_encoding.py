"""Causal numerical encodings of the dataset's stored CSR event frames.

Counts measure stored active-pixel observations, never recovered raw events.
Event coordinates remain in their own rectified camera; no RGB/GT warp occurs.
"""
from pathlib import Path
import numpy as np

ENCODINGS = {
    'count33': dict(channels=2, window_ns=33_000_000, kind='split_log_count'),
    'bins100': dict(channels=6, window_ns=100_000_000, kind='three_time_bins_split_log_count'),
    'surface20': dict(channels=2, window_ns=200_000_000, tau_ns=20_000_000, kind='split_last_observation'),
    'visualize_gray': dict(channels=1, kind='latest_stored_frame_background128_positive255_negative0'),
}


def fit_geometry(native_shape, size):
    """Pixel-centre transform for aspect-preserving area rasterization/padding."""
    height, width = map(int, native_shape)
    scale = min(size / width, size / height)
    nw, nh = round(width * scale), round(height * scale)
    left, top = (size - nw) // 2, (size - nh) // 2
    sx, sy = nw / width, nh / height
    transform = np.array([[sx, 0, left + (sx - 1) / 2],
                          [0, sy, top + (sy - 1) / 2], [0, 0, 1]], np.float64)
    return (nh, nw, top, left), transform


def calibration(archive):
    """Use the mutually consistent world-to-rectified-camera calibration pair."""
    get = lambda sensor, key: archive['rgb_calibration__' + sensor + '__' + key].astype(np.float64)
    rr, re = get('rgb', 'rotation_matrix'), get('event', 'rotation_matrix')
    tr, te = get('rgb', 'translation').reshape(3), get('event', 'translation').reshape(3)
    for rotation in (rr, re):
        assert np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-5)
        assert np.isclose(np.linalg.det(rotation), 1, atol=1e-5)
    relative_rotation = re @ rr.T
    relative_translation = te - relative_rotation @ tr
    return dict(K_rgb=get('rgb', 'intrinsic_matrix'), K_event=get('event', 'intrinsic_matrix'),
                R_rgb_to_event=relative_rotation, t_rgb_to_event=relative_translation)


class EventFrames:
    def __init__(self, times, indptr, indices, data, native_shape, size=224, count_cap=256):
        self.times = np.asarray(times, np.int64)
        self.indptr = np.asarray(indptr)
        self.indices = np.asarray(indices)
        self.positive = np.asarray(data) > 0
        self.native_shape = tuple(map(int, native_shape))
        self.size = size
        self.count_cap = count_cap
        self.geometry, self.pixel_transform = fit_geometry(self.native_shape, size)
        assert len(self.indptr) == len(self.times) + 1
        assert len(self.indices) == len(self.positive) == int(self.indptr[-1])
        assert len(self.times) and np.all(np.diff(self.times) >= 0)
        assert np.all(np.diff(self.indptr) >= 0) and int(self.indptr[0]) == 0

    @classmethod
    def load(cls, path, size=224, count_cap=256):
        with np.load(Path(path), allow_pickle=False) as z:
            value = cls(z['timestamp_ns'], z['indptr'], z['indices'], z['data'],
                        z['image_shape'], size=size, count_cap=count_cap)
            value.cameras = calibration(z)
            value.rgb_times = z['rgb__timestamp_ns'].astype(np.int64)
        return value

    def _pixels(self, indices):
        height, width = self.native_shape
        nh, nw, top, left = self.geometry
        y, x = np.divmod(indices, width)
        assert not len(indices) or (indices.min() >= 0 and indices.max() < height * width)
        yy = np.minimum(((y + .5) * nh / height).astype(np.int64), nh - 1) + top
        xx = np.minimum(((x + .5) * nw / width).astype(np.int64), nw - 1) + left
        return yy * self.size + xx

    def _counts(self, start, end):
        lo, hi = np.searchsorted(self.times, [start, end], side='right')
        a, b = map(int, self.indptr[[lo, hi]])
        pixels = self._pixels(self.indices[a:b])
        polarity = self.positive[a:b]
        channels = []
        for mask in (polarity, ~polarity):
            count = np.bincount(pixels[mask], minlength=self.size ** 2)
            channels.append(np.minimum(np.log1p(count) / np.log1p(self.count_cap), 1))
        return np.asarray(channels, np.float32).reshape(2, self.size, self.size)

    def encode(self, timestamp_ns, encodings=None):
        import cv2
        end = int(timestamp_ns)
        names = set(ENCODINGS) if encodings is None else set(encodings)
        assert names and names <= set(ENCODINGS)
        result = {}
        if 'count33' in names:
            result['count33'] = self._counts(end - 33_000_000, end)
        if 'bins100' in names:
            edges = end - 100_000_000 + np.arange(4, dtype=np.int64) * 100_000_000 // 3
            result['bins100'] = np.concatenate([self._counts(int(edges[i]), int(edges[i + 1]))
                                                for i in range(3)])
        if 'surface20' in names:
            lo, hi = np.searchsorted(self.times, [end - 200_000_000, end], side='right')
            a, b = map(int, self.indptr[[lo, hi]])
            pixels = self._pixels(self.indices[a:b]); polarity = self.positive[a:b]
            # Relative float64 times avoid subtracting ~1e18 ns as low-precision floats.
            observations = np.repeat((self.times[lo:hi] - end).astype(np.float64),
                                     np.diff(self.indptr[lo:hi + 1]).astype(np.int64))
            last = np.full((2, self.size ** 2), -np.inf)
            for channel, mask in enumerate((polarity, ~polarity)):
                np.maximum.at(last[channel], pixels[mask], observations[mask])
            result['surface20'] = np.exp(last / 20_000_000).astype(np.float32).reshape(2, self.size, self.size)
        latest = int(np.searchsorted(self.times, end, side='right')) - 1
        available = latest >= 0 and end - int(self.times[latest]) <= 20_000_000
        if 'visualize_gray' in names:
            # Match visualize_data.py at native pixels before area downsampling.
            gray = np.full(self.native_shape[0] * self.native_shape[1], 128, np.uint8)
            if available:
                start, stop = map(int, self.indptr[latest:latest + 2])
                gray[self.indices[start:stop]] = np.where(self.positive[start:stop], 255, 0)
            nh, nw, top, left = self.geometry
            padded = np.full((self.size, self.size), 128, np.uint8)
            padded[top:top + nh, left:left + nw] = cv2.resize(gray.reshape(self.native_shape),
                                                            (nw, nh), interpolation=cv2.INTER_AREA)
            result['visualize_gray'] = padded[None].astype(np.float32) / 255
        return result, dict(event_available=bool(available), latest_event_index=latest,
                            latest_event_timestamp_ns=int(self.times[latest]) if latest >= 0 else -1,
                            timestamp_ns=end)


def self_check():
    """Check actual causal boundaries, polarity preservation, decay, and gray parity."""
    origin = 1_700_000_000_000_000_000
    times = origin + np.array([0, 10_000_000, 20_000_000, 40_000_000], np.int64)
    value = EventFrames(times, np.arange(5), np.array([0, 0, 1, 2]),
                        np.array([127, -128, 127, 127], np.int8), (2, 2), size=2)
    encoded, metadata = value.encode(origin + 20_000_000)
    assert metadata['latest_event_index'] == 2 and metadata['event_available']
    assert encoded['count33'][0, 0, 0] > 0 and encoded['count33'][1, 0, 0] > 0
    assert encoded['count33'][:, 1, 0].sum() == 0, 'Future observation leaked'
    assert np.isclose(encoded['surface20'][0, 0, 0], np.exp(-1))
    assert np.array_equal(np.rint(encoded['visualize_gray'][0] * 255).astype(np.uint8),
                          np.array([[128, 255], [128, 128]], np.uint8))
    before, metadata = value.encode(origin - 1)
    assert not metadata['event_available']
    assert not before['count33'].any() and not before['surface20'].any()
    assert np.allclose(before['visualize_gray'], 128 / 255)
    boundary, _ = value.encode(origin + 33_000_000)
    assert boundary['count33'][0, 0, 0] == 0, 'Open left window boundary violated'
    assert set(encoded) == set(ENCODINGS)
    for name in ENCODINGS:
        single, _ = value.encode(origin + 20_000_000, encodings=[name])
        assert np.array_equal(single[name], encoded[name]), 'Online single-encoding/cache bundle mismatch'
    now = origin + 20_000_129
    precision = EventFrames(np.array([now - 1, now + 1]), np.arange(3), np.array([0, 1]),
                            np.array([127, 127], np.int8), (2, 2), size=2)
    exact, _ = precision.encode(now, encodings=['bins100'])
    assert exact['bins100'][:, 0, 0].sum() > 0
    assert exact['bins100'][:, 0, 1].sum() == 0, 'Integer-ns bin endpoint admitted a future observation'
    return dict(causal=True, polarity_separated=True, timestamp_precision=True,
                time_surface_decay=True, visualize_gray_parity=True, missing_coverage=True)


if __name__ == '__main__':
    print(self_check())
