import numpy as np

from keysg.scene_segmentor.floor_segmentation import FloorSegmentation

# Peak heights logged by the HM3DSem val runs, and the number of storeys in the GT.
CASES = {
    "00824": ([0.0, 2.22, 2.44], 1),
    "00843": ([0.0, 0.61, 2.56, 2.88, 3.32, 3.90, 5.72, 6.07], 2),
    "00861": ([-1.61, -1.13, -0.84, 0.83, 1.16, 1.71, 1.91, 3.76], 2),
    "00862": ([-3.24, -0.59, 0.0, 0.91, 2.73, 3.28, 3.76, 4.06, 5.71], 3),
    "00873": ([-3.07, -0.31, 0.02, 0.70, 0.95, 2.46], 2),
}


def test_every_storey_is_kept_without_gaps():
    for scene, (heights, n_floors) in CASES.items():
        edges = np.round(np.arange(-4.0, 7.0, 0.01), 2)
        peaks = np.searchsorted(edges, heights)
        hist = np.zeros(len(edges))
        hist[peaks] = np.linspace(1.0, 2.0, len(peaks))  # arbitrary but distinct peak strengths
        points = np.array([[0, min(heights) - 0.1, 0], [0, max(heights) + 0.1, 0]])

        bounds = FloorSegmentation(None)._cluster_peaks_to_boundaries(peaks, edges, hist, points)

        assert len(bounds) == n_floors, f"{scene}: {len(bounds)} floors, expected {n_floors}"
        for (_, top), (bottom, _) in zip(bounds, bounds[1:]):
            assert bottom - top < 1.0, f"{scene}: storey gap {top:.2f}..{bottom:.2f} is dropped"


if __name__ == "__main__":
    test_every_storey_is_kept_without_gaps()
    print("ok")
