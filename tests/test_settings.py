import shutil
import tempfile
from pathlib import Path

import pytest

from smartcam.settings import MASK, Settings

TEST_CONFIG = Path(__file__).parent / "config.yml"


def settings(extra: str = "") -> Settings:
    d = Path(tempfile.mkdtemp())
    shutil.copy(TEST_CONFIG, d / "smartcam.yml")
    if extra:
        (d / "smartcam.yml").write_text((d / "smartcam.yml").read_text() + extra)
    return Settings(d / "smartcam.yml")


def test_read_masks_literal_secrets_but_not_env_references():
    s = settings()
    view = s.read()
    assert view["mqtt"]["password"] == MASK                       # literal "pass" in the fixture
    assert view["cameras"]["front_door"]["record"] == f"rtsp://user:{MASK}@{{ip}}:554/live"


def test_unchanged_save_keeps_the_file_and_its_comments():
    s = settings()
    before = s.path.read_text()
    s.write(s.read())
    assert s.path.read_text() == before


def test_edit_changes_only_that_value_and_keeps_secrets():
    s = settings()
    view = s.read()
    view["rules"]["notify_score"] = 5
    view["cameras"]["driveway"]["location"] = "indoor"
    s.write(view)
    text = s.path.read_text()
    assert "notify_score: 5" in text
    assert "password: pass" in text                               # the masked value was kept
    assert "&dahua_main" in text and "*dahua_main" in text        # anchors survive
    assert "# Test fixture" in text                               # comments survive
    assert "event_url: null" in text                              # not an empty value


def test_cameras_can_be_added_and_removed():
    s = settings()
    view = s.read()
    del view["cameras"]["driveway"]
    view["cameras"]["garage"] = {"location": "outdoor", "ip": "192.0.2.30",
                                 "record": "rtsp://u:p@{ip}/main", "detect": "rtsp://u:p@{ip}/sub"}
    s.write(view)
    assert list(Settings(s.path).read()["cameras"]) == ["front_door", "living_room", "garage"]


def test_invalid_config_is_rejected_and_the_file_untouched():
    s = settings()
    before = s.path.read_text()
    view = s.read()
    view["cameras"]["front_door"]["location"] = "attic"
    with pytest.raises(ValueError):
        s.write(view)
    assert s.path.read_text() == before


def test_edited_zone_is_written_on_one_line():
    s = settings()
    view = s.read()
    view["cameras"]["front_door"]["zones"]["door"] = [0.5, 0.5, 1, 0.5, 1, 1]
    s.write(view)
    assert "door: [0.5, 0.5, 1, 0.5, 1, 1]" in s.path.read_text()
