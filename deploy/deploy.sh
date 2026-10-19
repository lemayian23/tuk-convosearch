#!/usr/bin/env bash
# =====================================================================
# deploy/deploy.sh — build, start, and health-gate the stack.
#
#   ./deploy/deploy.sh              build if needed, up -d, wait until ready
#   ./deploy/deploy.sh --rebuild    force a clean image build (no cache)
#   ./deploy/deploy.sh --logs       tail the last 200 lines of all services
#   ./deploy/deploy.sh --status     container states + /api/health as JSON
#   ./deploy/deploy.sh --no-gate    start everything and exit immediately
#
# Run it from the repo root as the `deploy` user (it must be in the docker
# group) — it finds the repo root from its own location, so this also works:
#
#   cd /opt/tuk-convosearch && ./deploy/deploy.sh
#
# THE HEALTH GATE IS THE POINT OF THIS SCRIPT. `docker compose up -d` returns
# as soon as containers are *created*; that says nothing about whether the
# embedder loaded, whether the corpus is indexed, or whether Ollama actually
# has the model. This script waits on /api/health (the readiness endpoint, see
# app/api/health.py) for up to 120 s, prints the individual checks, and if the
# app cannot become ready it restores the previous image instead of leaving you
# with a broken site and a confusing error page.
#
# We reach the app with `docker compose exec`, not curl on the host, because
# port 8000 is deliberately NOT published (see docker-compose.yml). If you want
# to test from the host you have to go through Caddy, i.e. the real URL.
# =====================================================================
set -euo pipefail

BLUE='\033[1;34m'; GREEN='\033[1;32m'; YELLOW='\033[1;33m'; RED='\033[1;31m'; OFF='\033[0m'
say() { printf '\n%b==> %s%b\n' "$BLUE" "$*" "$OFF"; }
ok()  { printf '%b    %s%b\n' "$GREEN" "$*" "$OFF"; }
warn(){ printf '%b    %s%b\n' "$YELLOW" "$*" "$OFF"; }
die() { printf '%b    %s%b\n' "$RED" "$*" "$OFF" >&2; exit 1; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
COMPOSE_FILE="$SCRIPT_DIR/docker-compose.yml"
ENV_FILE="$REPO_ROOT/.env"
APP_IMAGE="tuk-convosearch-app:latest"
ROLLBACK_IMAGE="tuk-convosearch-app:rollback"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-120}"
HEALTH_INTERVAL=5

MODE="deploy"
FORCE_REBUILD=0
for arg in "$@"; do
	case "$arg" in
		--rebuild)  FORCE_REBUILD=1 ;;
		--logs)     MODE="logs" ;;
		--status)   MODE="status" ;;
		--no-gate)  NO_GATE=1 ;;
		-h|--help)  sed -n '2,30p' "$0"; exit 0 ;;
		*)          die "Unknown option: $arg  (try --rebuild, --logs, --status, --no-gate)" ;;
	esac
done

[[ -f "$COMPOSE_FILE" ]] || die "Missing $COMPOSE_FILE"

# THE CRUCIAL FLAGS. Compose resolves every relative path (build context,
# ./data, ./corpus, env_file: .env) against the PROJECT DIRECTORY, which
# defaults to the folder holding the compose file — i.e. deploy/. Overriding it
# to the repo root is what makes the compose file's `context: .` mean "the repo"
# and what makes the repo-root .env the one that gets read.
compose() {
	docker compose --project-directory "$REPO_ROOT" --env-file "$ENV_FILE" -f "$COMPOSE_FILE" "$@"
}

cd "$REPO_ROOT"

# ------------------------------------------------------------------ #
# Convenience modes: no building, no health gate.
# ------------------------------------------------------------------ #
if [[ "$MODE" == "status" ]]; then
	say "Container status"
	compose ps
	say "Readiness (/api/health)"
	if compose exec -T app python -c '
import json, urllib.request
print(json.dumps(json.load(urllib.request.urlopen("http://127.0.0.1:8000/api/health", timeout=10)), indent=2))
' 2>/dev/null; then
		:
	else
		warn "The app container is not answering. Try: ./deploy/deploy.sh --logs"
	fi
	exit 0
fi

if [[ "$MODE" == "logs" ]]; then
	say "Last 200 lines, all services (Ctrl-C to stop following)"
	compose logs --tail=200
	say "Follow? Ctrl-C to leave"
	compose logs -f --tail=50
	exit 0
fi

# ------------------------------------------------------------------ #
# Preflight
# ------------------------------------------------------------------ #
say "Preflight"
command -v docker >/dev/null 2>&1 || die "docker not installed. Run deploy/bootstrap.sh as root first."
docker compose version >/dev/null 2>&1 || die "Compose plugin missing. sudo apt-get install -y docker-compose-plugin"
docker info >/dev/null 2>&1 || die "Cannot talk to the Docker daemon. Log out and back in (the bootstrap added you to the docker group), or run with sudo."
[[ -f "$ENV_FILE" ]] || die "No $ENV_FILE. Copy deploy/.env.example to .env and fill it in (see RUNBOOK.md step 4)."

# Read one key out of .env without assuming it is exported anywhere.
# Rules: last assignment wins, leading whitespace/`export` tolerated, an inline
# comment is only stripped when whitespace precedes the `#` (so a password
# containing # survives), and surrounding quotes are removed.
get_env() {
	sed -n "s/^[[:space:]]*\\(export[[:space:]]\\+\\)\\{0,1\\}$1[[:space:]]*=[[:space:]]*//p" "$ENV_FILE" |
		tail -n 1 |
		sed -e 's/[[:space:]]\{1,\}#.*$//' -e 's/^"\(.*\)"$/\1/' -e "s/^'\(.*\)'\$/\1/"
}

ENVIRONMENT="$(get_env TUK_ENVIRONMENT)"
SECRET_KEY_VALUE="$(get_env TUK_SECRET_KEY)"
OLLAMA_HOST="$(get_env TUK_OLLAMA_HOST)"
SITE_ADDRESS="$(get_env SITE_ADDRESS)"
[[ -n "$SITE_ADDRESS" ]] || SITE_ADDRESS="ask.lemayian.com"
ok ".env found: TUK_ENVIRONMENT=${ENVIRONMENT:-<unset>} SITE_ADDRESS=$SITE_ADDRESS"

# Only ever report the LENGTH of the secret, never the value.
if [[ "$ENVIRONMENT" == "production" && ${#SECRET_KEY_VALUE} -lt 32 ]]; then
	# The app itself refuses to boot in this state (the _check_production_safety
	# validator in app/config.py); failing here saves a restart loop and prints
	# the reason.
	die "TUK_SECRET_KEY is ${#SECRET_KEY_VALUE} characters, needs >= 32. Generate one:  openssl rand -hex 32"
fi
if [[ "$OLLAMA_HOST" == "http://localhost:11434" || "$OLLAMA_HOST" == "http://127.0.0.1:11434" ]]; then
	die "TUK_OLLAMA_HOST points at localhost, which inside the app container is the app itself. In compose it must be: http://ollama:11434"
fi

# The two bind-mounted directories must exist AND be writable by the app's
# non-root uid (10001, created in deploy/Dockerfile). Compose would otherwise
# have Docker create them owned by root and the app fails on first write with a
# confusing "attempt to write a readonly database".
say "Data directories"
mkdir -p "$REPO_ROOT/data" "$REPO_ROOT/corpus/tuk"
if [[ "$(id -u)" -eq 0 ]]; then
	chown -R 10001:10001 "$REPO_ROOT/data" "$REPO_ROOT/corpus"
	ok "data/ and corpus/ owned by uid 10001 (the app user)"
elif [[ -w "$REPO_ROOT/data" && -w "$REPO_ROOT/corpus" ]]; then
	ok "data/ and corpus/ already writable by $(id -un)"
else
	warn "data/ and corpus/ are not writable by $(id -un). The app runs as uid 10001 and cannot index."
	die "Fix ownership once:  sudo chown -R 10001:10001 $REPO_ROOT/data $REPO_ROOT/corpus"
fi

if [[ ! -f "$REPO_ROOT/web/dist/index.html" || ! -f "$REPO_ROOT/admin/dist/index.html" ]]; then
	warn "No web/dist and/or admin/dist bundle. The API works, but / redirects to"
	warn "/docs and /admin 404s. Build on your laptop and rsync up (RUNBOOK.md"
	warn "step 6); no compose change is needed for that."
fi

# ------------------------------------------------------------------ #
# Rollback snapshot: tag the image that is currently running.
# ------------------------------------------------------------------ #
if docker image inspect "$APP_IMAGE" >/dev/null 2>&1; then
	docker tag "$APP_IMAGE" "$ROLLBACK_IMAGE"
	ok "current image tagged as $ROLLBACK_IMAGE (rollback point)"
else
	warn "First build: no previous image, so no rollback point exists yet."
fi

# ------------------------------------------------------------------ #
# Build
# ------------------------------------------------------------------ #
say "Building the app image (context: repo root, dockerfile: deploy/Dockerfile)"
if [[ "$FORCE_REBUILD" -eq 1 ]]; then
	# --no-cache is slow (re-installs torch + re-downloads nothing, the baked
	# MiniLM weights come from HF) but it is the only way to shake a stale
	# layer after you edit requirements.txt.
	compose build --no-cache app
else
	compose build app
fi
ok "image built"

# ------------------------------------------------------------------ #
# Up
# ------------------------------------------------------------------ #
say "Starting ollama -> app -> caddy"
compose up -d
ok "containers created; compose will keep them restarting on boot (restart: unless-stopped)"

if [[ "${NO_GATE:-0}" == "1" ]]; then
	warn "--no-gate: skipping the readiness wait. Check with ./deploy/deploy.sh --status"
	compose ps
	exit 0
fi

# ------------------------------------------------------------------ #
# Health gate
# ------------------------------------------------------------------ #
# Poll /api/health through the app container itself. It is cached for 10s
# inside the app (see _CACHE_SECONDS in app/api/health.py), which is why the
# poll interval is 5s — faster adds nothing.
say "Waiting up to ${HEALTH_TIMEOUT}s for readiness"

readiness_reader='
import json, sys, urllib.request
try:
    d = json.load(urllib.request.urlopen("http://127.0.0.1:8000/api/health", timeout=10))
except Exception as exc:
    print("not-answering|(" + type(exc).__name__ + ") " + str(exc)); sys.exit(2)
c = d.get("checks", {})
print("%s|%s|chunks=%s documents=%s|ollama=%s model=%s error=%s" % (
    d.get("status", "?"),
    " ".join("%s=%s" % (k, v) for k, v in sorted(c.items())),
    d.get("chunks"), d.get("documents"),
    (d.get("ollama") or {}).get("host"),
    (d.get("ollama") or {}).get("expected_model"),
    (d.get("ollama") or {}).get("error"),
))
sys.exit(0 if d.get("status") == "ready" else 3)
'

deadline=$((SECONDS + HEALTH_TIMEOUT))
attempt=0
last=""
while [[ $SECONDS -lt $deadline ]]; do
	attempt=$((attempt + 1))
	if line="$(compose exec -T app python -c "$readiness_reader" 2>/dev/null)"; then
		say "Readiness after $((attempt * HEALTH_INTERVAL))s"
		printf '    status=ready  %s\n' "$line"
		ready=1
		break
	else
		rc=$?
		last="$line"
		if [[ $rc -eq 3 ]]; then
			printf '    [%3ds] degraded  %s\n' "$((attempt * HEALTH_INTERVAL))" "$last"
		else
			printf '    [%3ds] app not answering yet (container starting, or not up)\n' "$((attempt * HEALTH_INTERVAL))"
		fi
		ready=0
		sleep "$HEALTH_INTERVAL"
	fi
done

if [[ "${ready:-0}" -ne 1 ]]; then
	printf '\n%b========== DEPLOY DID NOT BECOME READY ==========%b\n' "$RED" "$OFF"
	printf 'Last readiness line: %s\n\n' "${last:-<none>}"
	say "App container state"
	compose ps || true
	say "App logs (last 60 lines)"
	compose logs --tail=60 app || true
	say "What the failure usually means"
	cat <<'DIAG'
  checks.model_present=False  -> the model is not downloaded. Run:
                                 ./deploy/pull-models.sh
  checks.ollama_reachable=False -> TUK_OLLAMA_HOST is wrong, or the ollama
                                 container is down:  ./deploy/deploy.sh --status
  checks.corpus_indexed=False   -> nothing indexed yet. Drop files in corpus/tuk
                                 and trigger a re-ingest (RUNBOOK.md step 12), or
                                 upload one through the admin panel.
  "app not answering yet"       -> look at the logs above. A ValueError about
                                 TUK_SECRET_KEY means .env is not production-safe.
                                 An embedder load can simply take > 120 s on a
                                 2 vCPU box: re-run with a longer budget,
                                 HEALTH_TIMEOUT=300 ./deploy/deploy.sh
DIAG
	if docker image inspect "$ROLLBACK_IMAGE" >/dev/null 2>&1; then
		say "Rolling back to the previous image"
		docker tag "$ROLLBACK_IMAGE" "$APP_IMAGE"
		compose up -d --no-build app
		warn "Started the PREVIOUS image. The new code is still in your working tree;"
		warn "fix .env or pull the model, then run ./deploy/deploy.sh again."
		warn "If the old image also fails, the problem is not your code: ./deploy/deploy.sh --logs"
	else
		warn "No rollback image exists (first deploy). The stack is left running so you"
		warn "can read its logs. To stop everything: compose down (see Makefile: make down)."
	fi
	exit 1
fi

# ------------------------------------------------------------------ #
# Caddy / TLS is the part humans forget to check.
# ------------------------------------------------------------------ #
say "TLS check through Caddy"
if curl -fsS --max-time 15 "https://$SITE_ADDRESS/api/health" >/dev/null 2>&1; then
	printf '    https://%s/api/health -> ready\n' "$SITE_ADDRESS"
else
	warn "https://$SITE_ADDRESS/api/health did not answer from here. That is EXPECTED"
	warn "if DNS for $SITE_ADDRESS does not point at this VPS yet, or if you are"
	warn "testing with a placeholder domain. Let's Encrypt can only validate a public"
	warn "hostname that resolves to this machine. Caddy retries the challenge every"
	warn "10 minutes; watch it with:"
	printf '    docker compose --project-directory %s --env-file %s -f %s logs -f caddy\n' \
		"$REPO_ROOT" "$ENV_FILE" "$COMPOSE_FILE"
	printf '    curl -I https://%s/api/health\n' "$SITE_ADDRESS"
fi

say "Deploy complete"
compose ps
cat <<DONE

Next steps
  Admin panel : https://$SITE_ADDRESS/admin
  API docs    : https://$SITE_ADDRESS/docs
  First admin : docker compose --project-directory $REPO_ROOT --env-file $ENV_FILE -f $COMPOSE_FILE \\
                  exec app python scripts/create_admin.py --email you@tuk.ac.ke
  Nightly backup: sudo install -m 644 deploy/tuk-backup.cron /etc/cron.d/tuk-backup
                  ./deploy/backup.sh   # run it once by hand to prove it works
DONE
