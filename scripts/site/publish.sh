#!/usr/bin/env bash
# Export recorded runs into site/ and deploy it to Vercel.
#
# Usage: scripts/site/publish.sh [--deploy] <cell-root>[:<run-id>[:<title>]] ...
#
# Each argument is a project or cell root (the directory holding automil/),
# optionally followed by the run id and title to show on the site. Cell roots
# copied from a cluster carry stamps in that host's zone: set SITE_TZ to its
# IANA name (default America/Vancouver, the campaign host). Without --deploy
# the site is only written, for a look with `python -m http.server -d site`.
set -euo pipefail
HERE="$(cd "$(dirname "$0")/../.." && pwd)"
SITE="$HERE/site"
TZ_NAME="${SITE_TZ:-America/Vancouver}"
deploy=0
if [ "${1:-}" = "--deploy" ]; then deploy=1; shift; fi
[ $# -gt 0 ] || { echo "usage: $0 [--deploy] <cell-root>[:<run-id>[:<title>]] ..." >&2; exit 2; }

rm -rf "$SITE/record" "$SITE/static" "$SITE/index.html"
for spec in "$@"; do
    IFS=':' read -r root run_id title <<< "$spec"
    args=(--out "$SITE" --tz "$TZ_NAME" --force)
    [ -n "${run_id:-}" ] && args+=(--run-id "$run_id")
    [ -n "${title:-}" ] && args+=(--title "$title")
    (cd "$HERE" && uv run automil --project "$root" viz export "${args[@]}")
done
echo "site written to $SITE ($(du -sh "$SITE" | cut -f1))"
if [ $deploy = 1 ]; then
    command -v vercel >/dev/null || { echo "vercel CLI not found: npm i -g vercel, then vercel login" >&2; exit 1; }
    (cd "$SITE" && vercel deploy --prod)
fi
