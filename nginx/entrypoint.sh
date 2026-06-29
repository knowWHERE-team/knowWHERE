#!/usr/bin/env sh
set -eu

tpl=/etc/nginx/templates/default.conf.template
out=/etc/nginx/conf.d/default.conf

if [ -n "${WEB_PROXY_API_KEY:-}" ]; then
  inject="    proxy_set_header x-api-key \"${WEB_PROXY_API_KEY}\";"
else
  inject=""
fi

sed "s|#INJECT_API_KEY#|${inject}|" "$tpl" > "$out"
exec nginx -g 'daemon off;'
