#!/usr/bin/env bash
# =====================================================================
# deploy/restore.sh — restore a tuk-backup-*.tar.gz onto THIS server.
#
#   ./deploy/restore.sh /var/backups/tuk-convosearch/tuk-backup-20261020-033000Z.tar.gz
#   ./deploy/restore.sh --latest
#   ./deploy/restore.sh --list
#
# What it does, in order:
#   1. stops the app (so nothing is writing to the database mid-restore)
#   2. moves the current data/ and corpus/ aside into a timestamped
#      .before-restore directory — it never deletes what is there now
#   3. unpacks the archive into place, deletes any stale -wal/-shm sidecars
#   4. chowns everything to uid 10001 (the app user inside the container)
#   5. restarts the stack and waits for /api/health to come back
#
# WHY THE WAL SIDECARS MUST GO
#   SQLite pairs a database file with its own tuk_convosearch.db-wal and
#   .db-shm. If a leftover WAL from the *old* database sits next to the
#   *restored* one, SQLite will try to apply those pages to a file they do not
#   belong to — the classic "database disk image is malformed" after a restore.
#   The snapshot inside the archive is already self-consistent (it was produced
#   by the online backup API), so the sidecars are pure trash.
#
# This restores data only. It does not pull models, recreate .env, or install
# Docker — if you are rebuilding the VPS from scratch, do RUNBOOK.md steps
# 1-5 first, then run this.
# =====================================================================
set -euo pipefail

BLUE='\033[1;34m'; GREEN='\033[1;32m'; YELLOW='\033[1;33m'; RED='\033[1;31m'; OFF='\033[0m'
say()  { printf '\n%b==> %s%b\n' "$BLUE" "$*" "$OFF"; }
ok()   { printf '%b    %s%b\n' "$GREEN" "$*" "$OFF"; }
warn() { printf '%b    %s%b\n' "$YELLOW" "$*" "$OFF"; }
die()  { printf '%b    %s%b\n' "$RED" "$*" "$OFF" >&2; exit 1; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
COMPOSE_FILE="$SCRIPT_DIR/docker-compose.yml"
ENV_FILE="$REPO_ROOT/.env"
BACKUP_DIR="${BACKUP_DIR:-/var/backups/tuk-convosearch}"

compose() {
	docker compose --project-directory "$REPO_ROOT" --env-file "$ENV_FILE" -f "$COMPOSE_FILE" "$@"
}

[[ -f "$ENV_FILE" ]] || die "No $ENV_FILE. Restore .env from your password manager first — it is not inside the backup."
cd "$REPO_ROOT"

# ------------------------------------------------------------------ #
# Which archive?
# ------------------------------------------------------------------ #
case "${1:-}" in
	--list)
		say "Backups available in $BACKUP_DIR (newest first)"
		ls -1ht "$BACKUP_DIR"/tuk-backup-*.tar.gz 2>/dev/null | sed 's/^/    /' || warn "none found"
		exit 0
		;;
	--latest)
		ARCHIVE="$(ls -1t "$BACKUP_DIR"/tuk-backup-*.tar.gz 2>/dev/null | head -n1 || true)"
		[[ -n "$ARCHIVE" ]] || die "No tuk-backup-*.tar.gz in $BACKUP_DIR. Copy one from your laptop first:
       rsync -a ~/tuk-backups/ deploy@YOUR_VPS_IP:$BACKUP_DIR/"
		;;
	"")
		die "Usage: ./deploy/restore.sh /path/to/tuk-backup-<stamp>.tar.gz | --latest | --list"
		;;
	*)
		ARCHIVE="$1"
		;;
esac

[[ -f "$ARCHIVE" ]] || die "Not a file: $ARCHIVE"

say "Archive to restore"
printf '    %s (%s)\n' "$ARCHIVE" "$(du -h "$ARCHIVE" | cut -f1)"
# Show the manifest before touching anything, so you can see which server and
# which day you are about to overwrite.
tar -xzOf "$ARCHIVE" MANIFEST.txt 2>/dev/null | sed 's/^/    | /' || warn "no MANIFEST.txt inside; is this one of our backups?"
printf '    Members: '; tar -tzf "$ARCHIVE" | sed -n '1,3p' | tr '\n' ' '; echo

say "This REPLACES the live database and corpus. Continue? Type yes"
read -r confirm
[[ "$confirm" == "yes" ]] || die "Aborted. Nothing was changed."

WORK="$(mktemp -d)"
STAMP="$(date -u +%Y%m%d-%H%M%SZ)"
trap 'rm -rf "$WORK"' EXIT

# ------------------------------------------------------------------ #
# 1. Stop the writers
# ------------------------------------------------------------------ #
say "1/5 Stopping the app (ollama and caddy can keep running)"
compose stop app
ok "app stopped"

# ------------------------------------------------------------------ #
# 2. Extract and check
# ------------------------------------------------------------------ #
say "2/5 Extracting"
tar -xzf "$ARCHIVE" -C "$WORK"
[[ -f "$WORK/tuk_convosearch.db" ]] || die "The archive has no tuk_convosearch.db at its root. Unpack it by hand and inspect."
if command -v sqlite3 >/dev/null 2>&1; then
	CHECK="$(sqlite3 "$WORK/tuk_convosearch.db" 'PRAGMA integrity_check;')" \
		|| die "sqlite3 could not open the restored file."
	[[ "$CHECK" == "ok" ]] || die "Restored database fails integrity_check: $CHECK"
	ok "integrity_check: ok ($(sqlite3 "$WORK/tuk_convosearch.db" 'SELECT COUNT(*) FROM documents;') documents, $(sqlite3 "$WORK/tuk_convosearch.db" 'SELECT COUNT(*) FROM chunks;') chunks)"
else
	warn "sqlite3 CLI missing; cannot pre-verify. Install it (sudo apt-get install -y sqlite3)"
	warn "and re-run, or accept the risk and continue."
fi

# ------------------------------------------------------------------ #
# 3. Move the current state aside
# ------------------------------------------------------------------ #
say "3/5 Putting the restored files in place (current data kept as *.before-restore-$STAMP)"
if [[ -d "$REPO_ROOT/data" ]]; then
	mv "$REPO_ROOT/data" "$REPO_ROOT/data.before-restore-$STAMP"
	warn "previous data/ saved at data.before-restore-$STAMP"
fi
mkdir -p "$REPO_ROOT/data"
install -m 644 "$WORK/tuk_convosearch.db" "$REPO_ROOT/data/tuk_convosearch.db"
# Drop any sidecars that came with the old install (there should be none, since
# data/ was just moved) and any that might have been archived.
rm -f "$REPO_ROOT/data"/tuk_convosearch.db-wal "$REPO_ROOT/data"/tuk_convosearch.db-shm "$REPO_ROOT/data"/tuk_convosearch.db-journal

if [[ -d "$WORK/corpus" ]]; then
	if [[ -d "$REPO_ROOT/corpus" ]]; then
		cp -a "$REPO_ROOT/corpus" "$REPO_ROOT/corpus.before-restore-$STAMP"
		warn "previous corpus/ saved at corpus.before-restore-$STAMP"
	fi
	# Replace rather than merge: a merged corpus would resurrect documents the
	# operator deliberately deleted after this backup was taken.
	rm -rf "$REPO_ROOT/corpus"
	cp -a "$WORK/corpus" "$REPO_ROOT/corpus"
	ok "corpus restored ($(find "$REPO_ROOT/corpus" -type f | wc -l | tr -d ' ') files)"
else
	warn "No corpus/ in the archive; leaving the current one untouched."
fi

# ------------------------------------------------------------------ #
# 4. Ownership
# ------------------------------------------------------------------ #
say "4/5 Ownership for the non-root app user (uid 10001)"
if [[ "$(id -u)" -eq 0 ]]; then
	chown -R 10001:10001 "$REPO_ROOT/data" "$REPO_ROOT/corpus"
else
	# The app user owns these inside the container; from the host this needs
	# root. Asking for sudo here is intentional — it is the only privileged step.
	sudo chown -R 10001:10001 "$REPO_ROOT/data" "$REPO_ROOT/corpus"
fi
chmod 755 "$REPO_ROOT/data"
ok "data/ and corpus/ owned by 10001"

# ------------------------------------------------------------------ #
# 5. Start and verify
# ------------------------------------------------------------------ #
say "5/5 Starting the app and waiting for readiness"
compose up -d app
DEADLINE=$((SECONDS + 180))
until compose exec -T app python -c '
import json,urllib.request
d=json.load(urllib.request.urlopen("http://127.0.0.1:8000/api/health",timeout=10))
print("    status=%s checks=%s" % (d["status"], d["checks"]))
raise SystemExit(0 if d["status"]=="ready" else 3)
' 2>/dev/null; do
	[[ $SECONDS -lt $DEADLINE ]] || die "App did not become ready in 180 s. See: ./deploy/deploy.sh --logs"
	sleep 10
done

cat <<DONE

Restore finished.

  Check the numbers you expect in the admin panel: https://\$(grep -m1 SITE_ADDRESS= $ENV_FILE | cut -d= -f2)/admin
  Index status (chunk count should match the restored database):
    docker compose --project-directory $REPO_ROOT --env-file $ENV_FILE -f $COMPOSE_FILE \\
      exec app python -c "import json,urllib.request;print(json.load(urllib.request.urlopen('http://127.0.0.1:8000/api/health')))"

  If something is wrong, roll the restore back:
    docker compose --project-directory $REPO_ROOT --env-file $ENV_FILE -f $COMPOSE_FILE stop app
    rm -rf $REPO_ROOT/data && mv $REPO_ROOT/data.before-restore-$STAMP $REPO_ROOT/data
    docker compose --project-directory $REPO_ROOT --env-file $ENV_FILE -f $COMPOSE_FILE up -d app

  Delete the .before-restore directories once you are confident:
    rm -rf $REPO_ROOT/data.before-restore-$STAMP $REPO_ROOT/corpus.before-restore-$STAMP
DONE
