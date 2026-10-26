import numpy as np

from smartcam.ha.doorbell import join


def test_cameras_are_joined_side_by_side_at_one_height():
    a, b = np.full((1944, 2592, 3), 50, np.uint8), np.full((1296, 2304, 3), 200, np.uint8)
    out = join([a, b], 720)
    assert out.shape == (720, 960 + 6 + 1280, 3)
    assert out[0, 0, 0] == 50 and out[0, -1, 0] == 200
