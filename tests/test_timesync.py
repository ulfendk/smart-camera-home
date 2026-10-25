from smartcam.timesync import posix_tz


def test_posix_tz_from_the_zone_file():
    assert posix_tz("Europe/Copenhagen") == "CET-1CEST,M3.5.0,M10.5.0/3"
    assert posix_tz("No/Such_Zone") is None
