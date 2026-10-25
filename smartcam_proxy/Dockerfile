# Built by the Home Assistant Supervisor when the add-on is installed.
FROM ghcr.io/hassio-addons/base:21.0.9

RUN apk add --no-cache nginx

COPY rootfs /

LABEL \
    io.hass.name="smartcam proxy" \
    io.hass.description="Opens the smartcam web UI in the Home Assistant sidebar" \
    io.hass.type="addon" \
    org.opencontainers.image.source="https://github.com/ulfendk/smart-camera-home" \
    org.opencontainers.image.licenses="MIT"
