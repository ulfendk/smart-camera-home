"""Read and write the config file for the settings UI.

The file is edited round-trip (ruamel.yaml), so comments, anchors and ${VAR} references survive a save.
Values are shown as written, before ${VAR} expansion, so secrets kept in .env never reach the browser;
literal passwords and tokens are masked, and a masked value sent back means "unchanged".
"""

import io
import os
import re
import tempfile
from pathlib import Path

from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedSeq

from .config import load_config

MASK = "••••••"
SECRET_KEYS = {"password", "token"}
URL_PASSWORD = re.compile(r"^(\w+://[^:/@]+:)([^@]+)(@)")


def _yaml() -> YAML:
    y = YAML()
    y.preserve_quotes = True
    y.width = 4096
    y.representer.add_representer(type(None), lambda r, _: r.represent_scalar("tag:yaml.org,2002:null", "null"))
    return y


def _plain(node):
    """ruamel nodes -> plain dicts/lists, aliases expanded."""
    if isinstance(node, dict):
        return {str(k): _plain(v) for k, v in node.items()}
    if isinstance(node, list):
        return [_plain(v) for v in node]
    return node


def _is_literal(value) -> bool:
    return isinstance(value, str) and value != "" and "${" not in value


def mask(node, key: str = ""):
    if isinstance(node, dict):
        return {k: mask(v, k) for k, v in node.items()}
    if isinstance(node, list):
        return [mask(v) for v in node]
    if key in SECRET_KEYS and _is_literal(node):
        return MASK
    if isinstance(node, str) and (m := URL_PASSWORD.match(node)) and "${" not in m.group(2):
        return URL_PASSWORD.sub(rf"\g<1>{MASK}\g<3>", node)
    return node


def _unmask(new, old):
    """Put back secrets the browser only saw masked."""
    if new == MASK:
        return old
    if isinstance(new, str) and MASK in new and isinstance(old, str) and (m := URL_PASSWORD.match(old)):
        return new.replace(MASK, m.group(2))
    return new


def _flow(value):
    """Lists are written on one line ([x, y, ...]), like the hand-written zones and masks."""
    if isinstance(value, list):
        seq = CommentedSeq(_flow(v) for v in value)
        seq.fa.set_flow_style()
        return seq
    if isinstance(value, dict):
        return {k: _flow(v) for k, v in value.items()}
    return value


def merge(doc, new: dict, top: bool = False) -> bool:
    """Update the round-trip document in place, touching only what changed (keeps anchors and comments).

    Keys missing from `new` are deleted, except at the top level, where the file may hold helpers
    such as YAML anchor definitions that the UI never sees. Returns whether anything changed.
    """
    changed = False
    if not top:
        for key in [k for k in doc if k not in new]:
            del doc[key]
            changed = True
    for key, value in new.items():
        old = doc.get(key)
        if isinstance(value, dict) and isinstance(old, dict):
            changed |= merge(old, value)
        else:
            value = _unmask(value, old)
            if _plain(old) != value:
                doc[key] = _flow(value)
                changed = True
    return changed


class Settings:
    def __init__(self, path: Path):
        self.path = path

    def _load(self):
        return _yaml().load(self.path.read_text())

    def read(self) -> dict:
        return mask(_plain(self._load()))

    def validate(self, text: str) -> None:
        """Raise ValueError if the config would not load. Uses the same loader as startup."""
        from .events import RulesConfig
        from .lights.controller import LightsConfig
        from .power import WatchdogConfig

        with tempfile.NamedTemporaryFile("w", suffix=".yml", dir=self.path.parent, delete=False) as f:
            f.write(text)
        try:
            cfg = load_config(Path(f.name))
            RulesConfig(**cfg.rules)
            LightsConfig(**cfg.lights)
            WatchdogConfig(**cfg.watchdog)
        except Exception as e:
            raise ValueError(str(e)) from e
        finally:
            os.unlink(f.name)

    def write(self, new: dict) -> bool:
        """Save the edited config. Returns False if nothing changed (the file is left alone)."""
        doc = self._load()
        if not merge(doc, new, top=True):
            return False
        buf = io.StringIO()
        _yaml().dump(doc, buf)
        text = buf.getvalue()
        self.validate(text)
        backup = self.path.with_suffix(self.path.suffix + ".bak")
        backup.write_text(self.path.read_text())
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(text)
        os.chmod(tmp, self.path.stat().st_mode)
        tmp.replace(self.path)
        return True
