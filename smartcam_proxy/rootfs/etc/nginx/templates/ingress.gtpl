server {
    listen 8099 default_server;

    root            /dev/null;
    server_name     $hostname;

    location / {
        # Only Home Assistant's ingress gateway, which has already checked the user's login.
        allow   172.30.32.2;
        deny    all;

        proxy_pass {{ .server }};
        proxy_set_header X-Ingress-Path {{ .entry }};
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Real-IP $remote_addr;

        include /etc/nginx/includes/proxy_params.conf;
    }
}
