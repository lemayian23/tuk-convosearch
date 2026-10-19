#!/usr/bin/env bash
# =====================================================================
# deploy/backup.sh — nightly snapshot of the things that cannot be re-created.
#
#   ./deploy/backup.sh            take a backup now
#   BACKUP_DIR=/mnt/nfs/tuk ./deploy/backup.sh
#   KEEP_DAYS=30 ./deploy/backup.sh
#
# WHAT IS BACKED UP, AND WHY
#   data/tuk_convosearch.db   Every document row, every chunk with its vector,
#                             every session, message and query-log line. This
#                             single file IS the durable store — losing it
#                             means re-uploading the corpus and losing all
#                             usage history.
#   corpus/                   The source PDF/DOCX/TXT files. Re-embedding is
#                             cheap; getting the original Kenyan government
#                             PDFs back off university websites is not.
#
# WHAT IS DELIBERATELY NOT BACKED UP
#   * The `ollama_models` Docker volume (multiple GB of weights). It is a
#     byte-for-byte copy of something you can download again:
#     ./deploy/pull-models.sh. Backing it up would make the nightly job bigger
#     than the entire rest of the VPS, for zero recovery value. If your model
#     choice changes (a tag gets re-pushed upstream), re-pulling is exactly what
#     you want anyway.
#   * The `caddy_data` volume (TLS certificates + ACME account key). Small, but
#     re-issuing is free and instant; only worth saving if you fear Let's
#     Encrypt rate limits. `docker volume inspect tuk-caddy-data` tells you
#     where it lives if you decide otherwise.
#   * The Docker image itself (rebuild from git) and .env (keep THAT in your
#     password manager — it is the one file you cannot recover any other way).
#
# WHY `sqlite3 .backup` AND NOT `cp data/tuk_convosearch.db`
#   The database runs in WAL mode (app/services/database.py sets
#   `PRAGMA journal_mode = WAL`), so the most recent commits live in
#   tuk_convosearch.db-wal and the -shm index is mid-flight. Copying the main
#   file while the app is writing gives you a snapshot that is missing those
#   transactions, and often a file that fails its own integrity check because
#   the page header and the free-list no longer agree. Restoring it later is
#   how you discover the backup was never a backup.
#   SQLite's online backup API (what the `.backup` shell command and Python's
#   Connection.backup() use) copies page-by-page inside a read transaction, so
#   it tolerates concurrent writers and produces a self-consistent single file
#   with the WAL already merged in. That is the only correct way to back up a
#   live SQLite database. `VACUUM INTO` would also be safe; it is not used here
#   because it rewrites/reorders pages and takes longer on a big corpus, and we
#   want the job to finish before the next student wakes up.
#
#   We run the backup INSIDE the app container because that is where the file
#   is writable/readable by the app's uid (10001), and because the image ships
#   python3 with the sqlite3 module (it does not ship the sqlite3 CLI). The copy
#   out of the container is a plain byte stream, so the host never touches the
#   live database file.
#
# A BACKUP ON THE SAME DISK IS NOT A BACKUP. The last line this script prints
# is the rsync command that takes the file off the VPS. NovaHost snapshots are
# not a substitute: they vanish with the account.
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
KEEP_DAYS="${KEEP_DAYS:-14}"          # 14 daily copies; see the rotation note
STAMP="$(date -u +%Y%m%d-%H%M%SZ)"
TMP_CONTAINER_PATH="/tmp/tuk-backup-$STAMP.db"

compose() {
	docker compose --project-directory "$REPO_ROOT" --env-file "$ENV_FILE" -f "$COMPOSE_FILE" "$@"
}

# ------------------------------------------------------------------ #
# Preflight
# ------------------------------------------------------------------ #
command -v docker >/dev/null 2>&1 || die "docker not installed."
[[ -f "$ENV_FILE" ]] || die "No $ENV_FILE — compose cannot be run."
cd "$REPO_ROOT"

say "Checking the app container"
APP_CONTAINER="$(compose ps -q app 2>/dev/null || true)"
[[ -n "$APP_CONTAINER" ]] || die "The app container is not running, so there is nothing to back up. Start the stack with ./deploy/deploy.sh"

# Create the destination. /var/backups is root-owned on Ubuntu, so the first
# run needs sudo once; afterwards the directory belongs to whoever ran this.
if [[ ! -d "$BACKUP_DIR" ]]; then
	say "Creating $BACKUP_DIR"
	if [[ -w /var/backups || "$(id -u)" -eq 0 ]]; then
		mkdir -p "$BACKUP_DIR"
	else
		sudo mkdir -p "$BACKUP_DIR"
	fi
	chown "$(id -un):$(id -gn)" "$BACKUP_DIR" 2>/dev/null || sudo chown "$(id -un):$(id -gn)" "$BACKUP_DIR" || true
fi
chmod 750 "$BACKUP_DIR" 2>/dev/null || true
[[ -w "$BACKUP_DIR" ]] || die "$BACKUP_DIR is not writable by $(id -un). Fix: sudo chown $(id -un) $BACKUP_DIR"

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

# ------------------------------------------------------------------ #
# 1. Database, via SQLite's online backup API
# ------------------------------------------------------------------ #
say "1/4 Snapshotting the database (online backup API, inside the app container)"
# -T so python gets no TTY; the code is identical in effect to
#   sqlite3 data/tuk_convosearch.db ".backup /tmp/....db"
# but the image has no sqlite3 CLI, only the standard-library module.
compose exec -T app python - <<PYEOF || die "The in-container backup failed. Is the app busy/vacuuming? Retry once."
import os, sqlite3, sys
# Same default the app itself uses (Settings.data_dir in app/config.py); the
# env_file only sets TUK_DATA_DIR if you uncommented it.
data_dir = os.environ.get("TUK_DATA_DIR") or "/srv/app/data"
src = os.path.join(data_dir, "tuk_convosearch.db")
dst = "$TMP_CONTAINER_PATH"
if not os.path.exists(src):
    print("no database file yet at " + src + " - nothing to back up", file=sys.stderr)
    sys.exit(4)
src_conn = sqlite3.connect(src, timeout=30)
dst_conn = sqlite3.connect(dst)
src_conn.backup(dst_conn)   # online backup API: consistent copy, WAL merged in
dst_conn.close(); src_conn.close()
print("snapshot written to " + dst)
PYEOF

say "Copying the snapshot out of the container"
compose exec -T app cat "$TMP_CONTAINER_PATH" > "$WORK/tuk_convosearch.db" \
	|| die "Could not read the snapshot back out of the container."
compose exec -T app rm -f "$TMP_CONTAINER_PATH" || true

# Integrity-check the COPY, not the live file. This is the part that turns a
# "probably fine" backup into one you can trust at 3am.
if command -v sqlite3 >/dev/null 2>&1; then
	say "Verifying the snapshot with sqlite3"
	CHECK="$(sqlite3 "$WORK/tuk_convosearch.db" 'PRAGMA integrity_check; SELECT COUNT(*) FROM documents; SELECT COUNT(*) FROM chunks;' 2>&1 | head -3)" || \
		die "integrity_check failed: $CHECK"
	ok "integrity ok ($(echo "$CHECK" | head -1), $(echo "$CHECK" | sed -n 2p) documents, $(echo "$CHECK" | sed -n 3p) chunks)"
else
	warn "sqlite3 CLI not found on the host; skipping verification. Install it:"
	warn "  sudo apt-get install -y sqlite3"
fi

# ------------------------------------------------------------------ #
# 2. Corpus
# ------------------------------------------------------------------ #
say "2/4 Archiving the corpus"
CORPUS_DIR="$REPO_ROOT/corpus"
[[ -d "$CORPUS_DIR" ]] || warn "$CORPUS_DIR does not exist; no corpus to archive."

# ------------------------------------------------------------------ #
# 3. One gzip tarball: db + corpus + a manifest
# ------------------------------------------------------------------ #
say "3/4 Packing"
MANIFEST="$WORK/MANIFEST.txt"
{
	echo "TUK-ConvoSearch backup $STAMP"
	echo "app image : $(docker image inspect tuk-convosearch-app:latest --format '{{.Id}} {{.Created}}' 2>/dev/null || echo unknown)"
	echo "models    : $(compose exec -T ollama ollama list 2>/dev/null | tail -n +2 | awk '{printf "%s ", $1}') (NOT included in this archive; re-pull with ./deploy/pull-models.sh)"
	echo "contents  : tuk_convosearch.db (SQLite online-backup snapshot), corpus/ (source documents), MANIFEST.txt"
	echo "restore   : ./deploy/restore.sh $BACKUP_DIR/tuk-backup-$STAMP.tar.gz"
} > "$MANIFEST"

TARBALL="$BACKUP_DIR/tuk-backup-$STAMP.tar.gz"
TAR_ARGS=(-C "$WORK" tuk_convosearch.db MANIFEST.txt)
if [[ -d "$CORPUS_DIR" ]]; then
	# -C to the repo root puts the archive entries under `corpus/...`, which is
	# exactly the layout restore.sh expects.
	TAR_ARGS+=(-C "$(dirname "$CORPUS_DIR")" corpus)
fi
tar -czf "$TARBALL" "${TAR_ARGS[@]}"
ok "$TARBALL ($(du -h "$TARBALL" | cut -f1))"

# Never leave the plaintext snapshot lying around outside the tarball.
rm -f "$WORK/tuk_convosearch.db"

# ------------------------------------------------------------------ #
# 4. Rotation
# ------------------------------------------------------------------ #
say "4/4 Rotating (keeping the newest $KEEP_DAYS copies)"
# Keep by COUNT, not by age: a nightly job that is skipped for 20 days (a
# reboot, a cron outage) would delete everything under an age-based rule.
mapfile -t ALL < <(ls -1t "$BACKUP_DIR"/tuk-backup-*.tar.gz 2>/dev/null || true)
if ((${#ALL[@]} > KEEP_DAYS)); then
	for OLD in "${ALL[@]:KEEP_DAYS}"; do
		rm -f "$OLD"
		echo "    deleted $(basename "$OLD")"
	done
fi
ok "$(ls -1 "$BACKUP_DIR"/tuk-backup-*.tar.gz 2>/dev/null | wc -l | tr -d ' ') copy/copies retained in $BACKUP_DIR"

echo
say "Take it off this disk"
cat <<OFFSITE
  From your laptop (this is the step that makes it a real backup):

    rsync -a deploy@YOUR_VPS_IP:$BACKUP_DIR/ ~/tuk-backups/

  And verify you can actually open one, on your laptop:

    tar -tzf ~/tuk-backups/tuk-backup-$STAMP.tar.gz

  Install the nightly job:

    sudo install -m 644 deploy/tuk-backup.cron /etc/cron.d/tuk-backup
OFFSITE
