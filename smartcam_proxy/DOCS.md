# smartcam proxy

This add-on puts the smartcam web UI in the Home Assistant sidebar through ingress. Home Assistant
handles the login, and the UI works wherever Home Assistant does, including the companion app and
remote access. smartcam keeps running on its own machine; the add-on only forwards to it.

## Installation

1. Settings → Add-ons → Add-on Store → ⋮ → Repositories, and add
   `https://github.com/ulfendk/smart-camera-home`.
2. Install **smartcam proxy**.
3. On the Configuration tab, set `server` to your smartcam address.
4. Start the add-on and switch on "Show in sidebar".

## Configuration

```yaml
server: "http://192.168.1.10:8971"
```

`server` is the address of the smartcam web UI as Home Assistant sees it: `http://` or `https://`,
host and port. Live view goes through the same address, so nothing else has to be reachable.

## Credits

Adapted from the Frigate Proxy add-on by Blake Blackshear (MIT licence),
https://github.com/blakeblackshear/frigate-hass-addons.
