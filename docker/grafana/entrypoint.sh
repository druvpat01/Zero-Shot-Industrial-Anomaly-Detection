#!/bin/sh
# Render the checked-in dashboards against the provisioned datasource, then
# start Grafana.
#
# Why this exists
# ===============
# docker/grafana/dashboards/defect_detection.json is a Grafana *export*: it
# declares an `__inputs` block with a `DS_PROMETHEUS` datasource input, and
# every panel refers to `${DS_PROMETHEUS}`. That is the portable form — it is
# what lets somebody drop the file into any Grafana through Dashboards > Import
# and pick their own Prometheus — and it is worth keeping.
#
# File provisioning cannot answer that prompt. Nothing asks the operator which
# datasource to use, so `${DS_PROMETHEUS}` reaches the browser unresolved and
# every panel renders "Datasource ${DS_PROMETHEUS} was not found". So the input
# is bound here instead, to the uid the datasource provisioning file declares,
# and the rendered copy is written into Grafana's own data directory (the
# mounted source is read-only, which is the correct state for a file that lives
# in git).
#
# The alternative — editing the JSON to hardcode a uid — would make the file
# work in exactly one Grafana. This way it works in both.
set -eu

SOURCE_DIR=/etc/grafana/dashboards-src
TARGET_DIR=/var/lib/grafana/dashboards

# Must match `uid` in docker/grafana/provisioning/datasources/prometheus.yml.
DATASOURCE_UID=defect-prometheus

mkdir -p "$TARGET_DIR"

rendered=0
for source in "$SOURCE_DIR"/*.json; do
    [ -e "$source" ] || continue
    sed "s/\${DS_PROMETHEUS}/${DATASOURCE_UID}/g" "$source" >"$TARGET_DIR/$(basename "$source")"
    rendered=$((rendered + 1))
done

echo "grafana-entrypoint: rendered ${rendered} dashboard(s) -> ${TARGET_DIR} (datasource uid ${DATASOURCE_UID})"

# The image's own entrypoint. exec for the same reason as everywhere else: it
# has to be the process that gets SIGTERM.
exec /run.sh "$@"
