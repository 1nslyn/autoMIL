# The public site

`site/` is the deploy target for automil.org. Only `vercel.json` and this
file are tracked; the page, the static assets and the recorded runs are
written here by `scripts/site/publish.sh`, which exports each listed cell
root with `automil viz export` and deploys the directory with the Vercel CLI.
The site uses hash routes, so no rewrite rule is needed.
