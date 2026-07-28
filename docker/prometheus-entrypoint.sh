#!/bin/sh
# Render docker/prometheus.yml with the configured scrape interval, then start
# Prometheus.
#
# Why this exists
# ===============
# PROMETHEUS_SCRAPE_INTERVAL is in .env.example, and a setting in .env that
# nothing reads is worse than no setting at all. Prometheus does not expand
# environment variables in its config file — deliberately, on their part — and
# the config is bind-mounted read-only from the repo, so it cannot be edited in
# place. So the interval is substituted into a copy under /tmp, which is what
# Prometheus is then pointed at.
#
# The checked-in file keeps a real value (15s) rather than a placeholder, so
# `prometheus --config.file=docker/prometheus.yml` still works outside compose.
# This script only overwrites the line when the variable says something else.
set -eu

SOURCE=/etc/prometheus/prometheus.yml
RENDERED=/tmp/prometheus.yml
INTERVAL="${PROMETHEUS_SCRAPE_INTERVAL:-15s}"

# Basic regular expressions and escaped groups: this runs under busybox sed in
# the prom/prometheus image, which is not GNU sed and does not have to support
# -E. `^\( *\)` preserves the indentation, so the YAML stays valid whatever the
# key is nested under.
sed "s/^\( *\)scrape_interval:.*/\1scrape_interval: ${INTERVAL}/" "$SOURCE" >"$RENDERED"

# Prometheus refuses to start when scrape_timeout exceeds scrape_interval, so an
# interval below the configured 10 s timeout would turn a tuning knob into a
# crash loop. Clamp instead: a sub-10s scrape is a legitimate thing to ask for
# on a busy line, and the timeout has no business being the thing that forbids
# it.
case "$INTERVAL" in
    *s)
        seconds="${INTERVAL%s}"
        case "$seconds" in
            '' | *[!0-9]*) seconds="" ;;  # 1m30s, 500ms, or a typo: leave it alone
        esac
        if [ -n "$seconds" ] && [ "$seconds" -lt 10 ]; then
            sed "s/^\( *\)scrape_timeout:.*/\1scrape_timeout: ${INTERVAL}/" "$RENDERED" >"$RENDERED.tmp"
            mv "$RENDERED.tmp" "$RENDERED"
            echo "prometheus-entrypoint: clamped scrape_timeout to ${INTERVAL} (below the 10s default)"
        fi
        ;;
esac

echo "prometheus-entrypoint: scrape_interval=${INTERVAL}, config=${RENDERED}"

# exec, so Prometheus is PID 1 and receives the SIGTERM that `docker compose
# down` sends. A shell in between swallows it and the container is killed after
# the 10-second grace period instead, which for a TSDB means an unclean
# shutdown and a replayed WAL on every start.
exec /bin/prometheus --config.file="$RENDERED" "$@"
