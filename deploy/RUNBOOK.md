# TUK-ConvoSearch — deployment runbook

Putting the assistant on `https://ask.lemayian.com` at NovaHost, from a fresh
Ubuntu 24.04 VPS, by copy-pasting one block at a time.

**Who this is for.** You. You do not need to be a sysadmin. Every command is
given with what it should print, and anything that can lock you out is written
in capital letters.

**Total time.** About 90 minutes, most of it waiting for a model download.
You can stop at the end of any section and come back; nothing here is fragile
between sections.

---

## Before you start — have these four things

| You need | Where to get it |
|---|---|
| VPS public IPv4 | NovaHost dashboard, e.g. `41.90.x.x` |
| root password for the VPS | NovaHost welcome email |
| Access to `lemayian.com` DNS | Your registrar's control panel |
| This repo on your laptop | the folder you are reading this in |

Export the address once per shell so the copy-paste blocks below work as-is
(you must re-run this in every new terminal — it does not survive closing SSH):

```bash
VPS_IP=41.90.x.x          # <-- CHANGE to your real IP
```

Verify you can reach the box before doing anything else:

```bash
ssh root@$VPS_IP
```

Expected: a password prompt, then a shell whose prompt ends in `root@yourhostname:~#`.
Type `exit` to leave. If this fails, stop here and fix it with NovaHost support —
nothing below works without SSH.

---

## 1. What is being deployed, in one paragraph

Three Docker containers on one machine. `ollama` runs the language model and is
reachable only from inside a private Docker network (no published port).
`app` is your FastAPI code (built from `deploy/Dockerfile`); it reads and writes
`./data` (a single SQLite file that holds documents, text chunks, vectors,
sessions and the query log) and `./corpus` (the source PDFs/DOCXs), and it also
serves the student UI and this admin panel from the same origin so there is no
CORS to configure. `caddy` is the only container with ports open to the internet:
443 for HTTPS and 80 for redirects plus Let's Encrypt challenges. Students hit
`https://ask.lemayian.com` → Caddy → app → ollama.

Two rules follow from that: **never publish port 8000 or 11434**, and **the only
files you must back up are `data/tuk_convosearch.db` and `corpus/`**.

---

## 2. Install Docker, the firewall, swap (on the VPS, as root)

Run this from the **root SSH session**, from any directory:

```bash
cd /tmp && curl -fsSL -o bootstrap.sh https://raw.githubusercontent.com/lemayian23/tuk-convosearch/main/deploy/bootstrap.sh
```

If you have not pushed the repo to GitHub yet (most likely on day one), copy the
file from your laptop instead — run this on your **laptop**:

```bash
scp deploy/bootstrap.sh root@$VPS_IP:/tmp/bootstrap.sh
```

Then on the VPS:

```bash
bash /tmp/bootstrap.sh
```

Expected: a blue `==> n/8 ...` header per step, packages downloading, and three
questions it asks you:

1. `Create a 4096 MB swapfile at /swapfile now? [Y/n]` → **press Enter** (yes).
   A 4–8 GB VPS running a language model needs swap; without it the OOM killer
   randomly murders containers. Takes about 10 seconds.
2. `Enable the firewall now (allow 22, 80, 443; deny everything else)? [Y/n]` →
   **press Enter** (yes).
3. `Install the nightly backup job ...? [Y/n]` → **press Enter**, or `n` — the
   script installs it once `deploy/tuk-backup.cron` exists on disk, which happens
   in step 4, so you will get a second chance there.

Then check the firewall actually did what you wanted:

```bash
ufw status numbered
```

Expected — `Status: active`, then 22/80/443 `ALLOW IN` and **11434 and 8000
`DENY IN`**, with nothing else inbound:

```
Status: active

     To                         Action      From
     --                         ------      ----
[ 1] 22/tcp                    ALLOW IN    Anywhere
[ 2] 80/tcp                    ALLOW IN    Anywhere
[ 3] 443/tcp                   ALLOW IN    Anywhere
[ 4] 11434/tcp                 DENY IN     Anywhere
[ 5] 8000/tcp                  DENY IN     Anywhere
```

```bash
docker compose version
free -h | grep -i swap
```

Expected: `Docker Compose version v2.x` and a swap line showing `4.0Gi`, not `0B`.

**Now reboot** so the swapfile, sysctl and Docker daemon config are all
guaranteed to be the running ones, and log back in as root:

```bash
sudo reboot
# wait ~30 seconds, then:
ssh root@$VPS_IP
```

---

## 3. SSH keys, then turn off password login

Do this **now**, before deploying anything, while there is nothing on the server
to lose. Full explanation and the exact commands are printed at the end of
`bootstrap.sh`; they are repeated here so you do not have to scroll back.

On your **laptop**:

```bash
# Skip if ~/.ssh/id_ed25519.pub already exists:
ssh-keygen -t ed25519 -C "you@your-laptop"
ssh-copy-id -i ~/.ssh/id_ed25519.pub root@$VPS_IP
ssh deploy@$VPS_IP        # bootstrap created the 'deploy' user and copied the key
```

Expected: `deploy@ask:~$` with no password prompt. **Keep the original root
session open.**

On the VPS, as root — only after the key login above worked:

```bash
tee /etc/ssh/sshd_config.d/60-hardening.conf >/dev/null <<'SSHD'
PasswordAuthentication no
PubkeyAuthentication yes
KbdInteractiveAuthentication no
PermitRootLogin prohibit-password
MaxAuthTries 4
SSHD

sshd -T | grep -Ei 'passwordauthentication|permitrootlogin|maxauthtries'
```

Expected — this is the check, do not skip it:

```
passwordauthentication no
permitrootlogin prohibit-password
maxauthtries 4
```

Then, from a **new** laptop terminal, confirm `ssh deploy@$VPS_IP` still works,
and only then:

```bash
systemctl reload ssh
```

If you ever get locked out, the NovaHost control panel has a VNC/serial console
that does not use SSH. Log in as root there, delete
`/etc/ssh/sshd_config.d/60-hardening.conf`, run `systemctl reload ssh`.

---

## 4. Get the code onto the VPS

You will work as the `deploy` user from here on. Everything lands in
`/opt/tuk-convosearch`, which `bootstrap.sh` already created and owns.

From now on, **all commands run as `deploy` on the VPS** unless marked
`[laptop]`.

```bash
ssh deploy@$VPS_IP
cd /opt/tuk-convosearch
```

Option A — from GitHub (best: you can update later with one command):

```bash
git clone https://github.com/lemayian23/tuk-convosearch.git .
```

Option B — copy what is on your laptop right now. Run on **[laptop]**, note the
trailing slash on both sides:

```bash
rsync -av --exclude '.git' --exclude 'node_modules' --exclude '.venv' \
  --exclude 'admin/dist' --exclude 'web/dist' --exclude 'data' --exclude '.env' \
  ./ deploy@$VPS_IP:/opt/tuk-convosearch/
```

Expected: a file list ending in `total size is ... speedup is ...`.

Now create the config and the data directories:

```bash
cd /opt/tuk-convosearch
cp deploy/.env.example .env
chmod 600 .env
mkdir -p data corpus/tuk
sudo chown -R 10001:10001 data corpus
```

Why `10001`: that is the uid the container runs as (created in
`deploy/Dockerfile`). If `data/` and `corpus/` stay owned by `deploy`, the app
fails on its first write with `attempt to write a readonly database` — an error
that looks like a code bug and is not one.

Generate the secret and put it in the file:

```bash
openssl rand -hex 32
nano .env
```

In `nano` (Ctrl-W to search, Ctrl-O then Enter to save, Ctrl-X to exit) change
exactly two lines:

```
TUK_SECRET_KEY=<paste the 64 hex characters you just generated>
SITE_ADDRESS=ask.lemayian.com
```

Leave `TUK_ENVIRONMENT=production` and `TUK_OLLAMA_HOST=http://ollama:11434` as
they are. Leave both `TUK_BOOTSTRAP_ADMIN_*` lines empty for now — step 8 creates
the admin with a proper CLI instead.

```bash
grep -c '^' .env && grep '^SITE_ADDRESS=' .env
```

Expected: the line count, then `SITE_ADDRESS=ask.lemayian.com`.

> **Keep the secret in your password manager.** It is the one file that is not in
> any backup; without it every admin token is invalidated (annoying) but with it
> anyone can mint admin tokens (bad). `.env` is git-ignored, so it will not leak
> through `git push`.

---

## 5. Build the frontends and copy them up

The Docker image intentionally does **not** run npm (a RAM-starved data-centre box
is a bad place to run a JS build), and `deploy/docker-compose.yml` bind-mounts the
bundles from the host instead. So build them on your laptop.

**[laptop]**

```bash
npm --prefix admin ci && npm --prefix admin run build
npm --prefix web   ci && npm --prefix web   run build     # the student SPA
rsync -av admin/dist/ deploy@$VPS_IP:/opt/tuk-convosearch/admin/dist/
rsync -av web/dist/   deploy@$VPS_IP:/opt/tuk-convosearch/web/dist/
```

Expected for the admin build: `dist/index.html` and `dist/assets/*.js` listed, and
the last line says something like `✓ built in 3.2s`. The bundle is built with
`base: '/admin/'` (see `admin/vite.config.js`) so it works when served under
`/admin` — that is why the URLs come out as `/admin/assets/...`.

If `web/` does not exist in your checkout yet (the student UI is still being
written), that is fine: comment the `./web/dist` bind mount out in
`deploy/docker-compose.yml`, or leave it — an empty mount makes `/` return 404
instead of redirecting to `/docs`. Only `/admin` matters for now, and only
after step 8.

---

## 6. Pull the language model (this is the long one)

```bash
cd /opt/tuk-convosearch
./deploy/pull-models.sh
```

Expected: it reads `.env`, warns you about the download size, then shows Ollama's
own progress bar for `llama3.2:1b` (~1.3 GB), finishing with

```
verifying against Ollama's own /api/tags
    models visible to the app: ["llama3.2:1b:latest"]
    llama3.2:1b visible to the app
```

If it fails with a `no space left on device`, check `df -h /` — a partial download
leaves layers behind; `./deploy/pull-models.sh --list` shows what you have.

**Which model should you actually use?** Measure it on your box instead of
trusting anyone's blog:

```bash
./deploy/pull-models.sh --bench
```

This prints the decision table (1b vs 3b vs 7b: disk, RAM while serving,
tokens/sec on CPU, expected answer latency) and then times the same real question
through each installed model. Pick the winner, set `TUK_OLLAMA_MODEL` in `.env`,
re-run `./deploy/pull-models.sh`, then `./deploy/deploy.sh`. On an 8 GB VPS the
sweet spot is usually `llama3.2:3b` with `TUK_MAX_CONCURRENT_LLM_CALLS=2`.

---

## 7. Deploy, and prove it is healthy

```bash
./deploy/deploy.sh
```

What happens, in order: it validates `.env` (secret length, that
`TUK_OLLAMA_HOST` is `http://ollama:11434` and not `localhost`), checks directory
ownership, tags the current image as a rollback point, builds, starts
`ollama → app → caddy`, then polls `/api/health` every 5 s for up to 120 s
printing each attempt.

Expected last lines:

```
==> Readiness after 75s
    status=ready  corpus_indexed=True model_present=True ollama_reachable=True chunks=412 documents=7
==> TLS check through Caddy
    https://ask.lemayian.com/api/health -> ready
==> Deploy complete
```

If instead you see `DEPLOY DID NOT BECOME READY`, the script prints the app's logs
and a diagnosis list, then restarts the previous image. Read the printed
diagnosis; the short version is in the troubleshooting table at the end.

The first build takes 5–15 minutes (pip installs torch + sentence-transformers and
bakes the embedding model into the image). It is cached afterwards, so a code-only
change rebuilds in seconds.

Confirm the containers and their health:

```bash
./deploy/deploy.sh --status
```

Expected: three rows, all `Up`, `tuk-app` and `tuk-ollama` and `tuk-caddy` healthy,
then the full JSON from `/api/health`.

Also check from your laptop that nothing is exposed that should not be:

```bash
nc -zv $VPS_IP 443     # expect: succeeded / open
nc -zv $VPS_IP 8000    # expect: Connection refused or a timeout
nc -zv $VPS_IP 11434   # expect: Connection refused or a timeout
```

Those last two **must** fail. If `8000` answers, you added a `ports:` entry to the
app service — remove it.

---

## 8. Create the first admin, then log in

```bash
cd /opt/tuk-convosearch
docker compose --project-directory . --env-file .env -f deploy/docker-compose.yml \
  exec app python scripts/create_admin.py --email you@tuk.ac.ke
```

(Shorter: `make admin EMAIL=you@tuk.ac.ke` — the `Makefile` is at the repo root.)

Expected: it prompts once for the password (hidden input, minimum 12 characters —
and at most 72 bytes, because bcrypt silently truncates anything longer, so do not
paste a 100-character one) and prints the created user id, e.g.
`Created admin #1 for you@tuk.ac.ke.`. There is no confirmation prompt, so type it
carefully. Without any admin account, `/api/admin/*` returns 401 and the panel is
useless, so this step is not optional.

Now open **`https://ask.lemayian.com/admin`**, sign in, and click
**Index & Model**. You should see `Ollama reachable`, `model present`, a non-zero
chunk count, and your corpus files listed.

---

## 9. DNS — do this before you expect HTTPS to work

**Caddy cannot get a certificate until `ask.lemayian.com` resolves to this VPS.
Let's Encrypt validates the public address of a domain name: it cannot and will
not validate `localhost`, a bare IP, or a name still pointing at an old server.**

In your DNS control panel for `lemayian.com`, add one record:

| Type | Name/Host | Value | TTL |
|---|---|---|---|
| A | `ask` | your VPS public IPv4 | 300 |

Then, from your laptop, watch it propagate (a few minutes to a couple of hours;
`147.x.x.x` should equal your VPS IP):

```bash
dig +short ask.lemayian.com
# or, if dig is not installed:
nslookup ask.lemayian.com
```

Only once that returns the right IP will Caddy's certificate succeed. If Caddy
started before DNS was ready, it does **not** need restarting — it retries the
challenge roughly every 10 minutes on its own. Watch it happen:

```bash
docker compose --project-directory . --env-file .env -f deploy/docker-compose.yml logs -f caddy
```

Expected success lines:

```
acme: obtained certificate for ask.lemayian.com
serving initial configuration
```

Verify from the laptop:

```bash
curl -I https://ask.lemayian.com/api/health
```

Expected: `HTTP/2 200`, plus your security headers
(`strict-transport-security: max-age=31536000; includeSubDomains`,
`content-security-policy`, `x-content-type-options: nosniff`). A valid
`HTTP/2 200` over HTTPS means Let's Encrypt worked. If you get
`curl: (60) SSL certificate problem`, DNS has not propagated to the resolver your
laptop uses, or you are still pointing at a placeholder domain — wait and retry.

Also check the redirect: `curl -I http://ask.lemayian.com/api/health` must show
`308`/`301` to the `https://` URL, not a page served over plain HTTP.

---

## 10. Smoke test: ask it a real question

Blocking answer first (this is what the student UI falls back to):

```bash
curl -sS -X POST https://ask.lemayian.com/api/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"When do the second semester exams start?"}' | head -c 600
```

Expected: JSON with a `session_id`, an `answer`, `sources` naming a real document,
`"grounded": true`, and `"route": "internal_docs"`.

Streaming (what the UI uses — Server-Sent Events, so you should see lines appear
gradually rather than all at once):

```bash
curl -N -sS -X POST https://ask.lemayian.com/api/chat/stream \
  -H 'Content-Type: application/json' \
  -d '{"message":"What are the tuition fees for this year?"}'
```

Expected: a stream of `data: {...}` lines starting with
`{"type":"start","session_id":"..."}`, several token/delta events, ending with
`{"type":"end"}`. Keep the `-N` — it disables curl's own buffering.

Follow one conversation (note: session ids are 8–64 chars of letters/digits/`_`/`-`):

```bash
curl -sS "https://ask.lemayian.com/api/sessions?limit=5"
```

Expected: a JSON array of sessions with `session_id`, `title`, `message_count`.

Rate limiting is live: 28 rapid questions per minute from one IP returns `429`:

```bash
for i in $(seq 1 30); do curl -s -o /dev/null -w '%{http_code} ' -X POST \
  https://ask.lemayian.com/api/chat -H 'Content-Type: application/json' \
  -d '{"message":"hi"}'; done; echo
```

Expected: a row of `200`s then some `429`s. (Wait a minute before testing again.)

---

## 11. Day-2 operations

`make help` lists every shortcut. The commands below are the raw equivalents, so
they work even if you forget `make` exists.

Set this once per shell to shorten the Docker commands:

```bash
alias dc='docker compose --project-directory /opt/tuk-convosearch --env-file /opt/tuk-convosearch/.env -f /opt/tuk-convosearch/deploy/docker-compose.yml'
```

### Where the logs are

```bash
dc logs --tail=200 app        # the FastAPI app: requests, warnings, tracebacks
dc logs --tail=100 caddy      # every request + TLS/ACME events
dc logs --tail=100 ollama     # model loading, OOM and queue messages
dc logs -f                     # all of it, following
./deploy/deploy.sh --logs      # same thing, wrapper-aware
```

Logs are capped at 3 × 10 MB per container (`logging:` options in
`docker-compose.yml` plus a daemon-wide default set by `bootstrap.sh`), so they
can never fill the disk. Inside the containers they also live at
`/var/lib/docker/containers/*/*-json.log` if you ever need the raw file.

Questions students asked are in the database, not just the logs: the admin
panel's **Query Logs** tab, or `make sqlite`.

### Add a document

Either route works; use whichever is nearer.

*In the panel:* **Documents → Upload**, choose the PDF/DOCX/TXT/MD, press
*Upload and Index*. It embeds inside the request (about 1 s per chunk on CPU), so
big files take a while — keep the tab open. If you get the yellow/red box saying
"No readable text found in this file. A scanned PDF needs OCR first", the file is a
photo of pages: `sudo apt install ocrmypdf && ocrmypdf scan.pdf clean.pdf`, then
upload `clean.pdf`.

*From the command line* (no token needed, writes straight into the mounted corpus):

```bash
# [laptop] copy a new policy PDF into the corpus
scp new-calendar.pdf deploy@$VPS_IP:/opt/tuk-convosearch/corpus/tuk/
# [VPS] then index every file in that directory
cd /opt/tuk-convosearch
dc exec app python -c "from app.config import get_settings as g; from app.ingestion.index import get_index; print(get_index().ingest_path(g().corpus_dir/'new-calendar.pdf'))"
```

Expected: `{'status': 'indexed', 'filename': 'new-calendar.pdf', 'document_id': 8, 'chunks': 23}`.
(`status: unchanged` means the same content is already indexed — nothing to do.)
To index the whole directory at once, use the panel's **Re-index corpus** button
(which calls `POST /api/admin/documents/ingest-all`, returns `202` immediately and
runs in the background) or `make reindex`.

Delete a document from the panel's table (**Remove**) or
`DELETE /api/admin/documents/{document_id}` — that deactivates it, drops its
vectors, and removes the file so a later re-index does not resurrect it.

With a token (curl equivalent of the panel), for scripting:

```bash
TOKEN=$(curl -sS -X POST https://ask.lemayian.com/api/auth/login \
  -H 'Content-Type: application/json' \
  -d '{"email":"you@tuk.ac.ke","password":"..."}' | python3 -c 'import json,sys;print(json.load(sys.stdin)["access_token"])')

curl -sS "https://ask.lemayian.com/api/admin/documents?include_inactive=true" \
  -H "Authorization: Bearer $TOKEN" | python3 -m json.tool | head -30

curl -sS -X POST "https://ask.lemayian.com/api/admin/documents" \
  -H "Authorization: Bearer $TOKEN" -F "file=@./corpus/tuk/2026 Academic Year Calendar.docx"
```

The field name is exactly `file`. The endpoint returns `201` with
`{status, filename, document_id, chunks}`.

### Back up, and take it off the server

```bash
./deploy/backup.sh                     # snapshot + tar.gz into /var/backups/tuk-convosearch
ls -lh /var/backups/tuk-convosearch/    # expect: dated tuk-backup-*.tar.gz files
sudo install -m 644 deploy/tuk-backup.cron /etc/cron.d/tuk-backup   # nightly at 03:30 UTC
```

Expected: `integrity ok (N documents, M chunks)`, then the archive path with its
size, then the reminder to rsync it off-box. **A backup on the same disk is not a
backup** — from your laptop:

```bash
rsync -a deploy@$VPS_IP:/var/backups/tuk-convosearch/ ~/tuk-backups/
```

Only `data/tuk_convosearch.db` and `corpus/` are inside: the Ollama model volume is
deliberately excluded because `./deploy/pull-models.sh` reproduces it for free.
Keep `.env` in your password manager; it is in no backup.

### Restore

```bash
cd /opt/tuk-convosearch
./deploy/restore.sh --list
./deploy/restore.sh --latest          # or the exact archive path
```

It stops the app, saves the current `data/` and `corpus/` as
`*.before-restore-<stamp>` (never deletes them), installs the restored files,
removes stale `-wal`/`-shm` sidecars, fixes ownership, restarts and waits for
`ready`. Type `yes` when asked. If the result is wrong, the rollback it prints
puts the previous state back.

### Check disk and RAM

```bash
df -h /                               # disk: keep > 2 GB free or Docker breaks
free -h                               # RAM + swap
docker system df                        # images / containers / volumes / build cache
docker stats --no-stream                 # live per-container RAM (ollama should be ~2-4 GB)
docker ps --format 'table {{.Names}}\t{{.Status}}'   # Status must not say "Restarting"
```

Expected roughly: `tuk-ollama` using most memory, `tuk-app` 0.6–1.2 GB, `tuk-caddy`
under 50 MB. If `tuk-ollama` is at its 6 GB limit and the machine is swapping hard,
lower `TUK_MAX_CONCURRENT_LLM_CALLS` to 1 — that is the knob, not the memory limit.

Docker build cache is the usual disk hog:

```bash
docker builder prune -f          # safe: only removes build cache, not images
docker image prune -f            # safe: removes dangling (untagged) images only
```

Never run `docker system prune -a` here — it deletes the `ollama` image you do not
want to re-download, and can remove a stopped container you were inspecting.

### Update the code after a change

```bash
cd /opt/tuk-convosearch
git pull                # or rsync from your laptop, as in step 4
./deploy/deploy.sh --rebuild      # after touching requirements.txt or the Dockerfile
./deploy/deploy.sh                # code-only change: cached build, ~1 min
```

`deploy.sh` health-gates and rolls back automatically, so a bad build leaves the
old version serving students rather than leaving a broken site.

### Restart / stop / start

```bash
dc restart app          # pick up new code without touching ollama or certs
dc stop                 # stop everything (volumes and data survive)
dc up -d                # start everything again
sudo reboot             # safe: restart: unless-stopped brings all three back
```

---

## 12. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `/api/health` says `degraded`, `model_present: false` | The model named in `TUK_OLLAMA_MODEL` is not in the ollama volume. Often a typo or a tag that changed (`llama3.2:1b` vs `llama3.2:1b-instruct-q4_K_M`). | `./deploy/pull-models.sh --list` to compare against `.env`; then `./deploy/pull-models.sh`. Reboot app: `dc restart app`. |
| `ollama_reachable: false`, error `Connection refused` | `TUK_OLLAMA_HOST` is `http://localhost:11434` (inside the container localhost is the app itself) or the ollama container is down. | Set `TUK_OLLAMA_HOST=http://ollama:11434` in `.env`, then `dc up -d app`. |
| `corpus_indexed: false` while `documents > 0` | Files are in the database but the vector table is empty — the corpus directory was replaced without re-indexing. | Panel → **Re-index corpus**, or `make reindex`. |
| Chat returns `503 All 2 model slots are busy. Please try again in a moment.` | More concurrent questions than the box can generate. This is the app protecting itself from the OOM killer, not a bug. | Immediate: students retry in ~45 s. Real fix: lower `TUK_MAX_CONCURRENT_LLM_CALLS` to 1, or use a smaller `TUK_OLLAMA_MODEL`, or buy RAM. Raise it only with RAM to match (see `deploy/.env.example`). |
| Chat returns `504` / takes > 2 minutes | CPU inference on a model too big for the box, or `TUK_LLM_TIMEOUT_SECONDS` below real generation time. | `--bench` the models; drop to `llama3.2:1b`; keep `TUK_LLM_TIMEOUT_SECONDS` above real generation time. Caddy puts no deadline on proxied responses, so a slow answer is not cut off by the proxy — the app's own budget is what fires. |
| `docker compose ps` shows `tuk-caddy` as `Restarting`, while `https://…` fails and `./deploy/deploy.sh --status` still looks fine | Caddy declared one hostname twice — usually `SITE_ADDRESS` and `WWW_ADDRESS` set to the same value in `.env`. It exits during startup without writing an error line, so nothing is ever listening on 80/443. | `grep -E '^(SITE|WWW)_ADDRESS' .env`, give them different values, then `dc up -d caddy`. `deploy/deploy.sh` now refuses this pairing at preflight. |
| SSE stream shows the whole answer at once instead of typing out (but `curl -N` against the container works) | Something is buffering the response. Caddy's own buffering is disabled in `deploy/Caddyfile` (`flush_interval -1`); a CDN, a browser extension, or an added nginx layer in front is the usual culprit. | Confirm with `curl -N https://ask.lemayian.com/api/chat/stream`. The app sends `X-Accel-Buffering: no` for nginx; if you put Cloudflare in front, enable SSE support / disable caching for `/api/`. |
| `429 You are asking too quickly` for everyone at once, or students all rate-limited together | `TUK_TRUST_FORWARDED_FOR` is `false` (so every student counts as Caddy's IP), or one busy computer lab shares one public IP. | Keep `TUK_TRUST_FORWARDED_FOR=true` (safe only because 8000 is unpublished) and raise `TUK_RATE_LIMIT_PER_MINUTE`. |
| `413` when uploading a PDF | The file exceeds `TUK_MAX_UPLOAD_MB` (15) or Caddy's 20 MB `request_body` limit. | Shrink/split the PDF, or raise both together — Caddy's limit must stay above the app's. |
| Upload says "No readable text found in this file. A scanned PDF needs OCR first." (HTTP 422) | A scanned/image-only PDF. Extraction succeeded, there was nothing to read. | `ocrmypdf scan.pdf clean.pdf` then upload `clean.pdf`. |
| Browser shows a blank `/admin` page; console shows 404 for `/assets/index-*.js` | The bundle was built without `base: '/admin/'` (someone reverted `admin/vite.config.js`), so it asks for the wrong paths. | Rebuild and re-sync: `npm --prefix admin run build` on the laptop, then the `rsync` from step 5. |
| `curl https://ask.lemayian.com` fails with an SSL error, but the site works by IP | DNS not propagated, wrong A record, or Caddy is still retrying the ACME challenge. | `dig +short ask.lemayian.com`; `dc logs --tail=100 caddy` and look for `acme: error`. Wait for propagation; do not keep restarting Caddy — it self-retries. |
| Caddy log: `cannot issue certificate: DNS problem` | The name resolves somewhere else, or AAAA/old A record wins. | Point **only** the A record for `ask` at the VPS; delete stale records; TTL 300 speeds this up. |
| Let's Encrypt fails with `too many failed authorizations` | Rate limit from repeated challenges (worst with `localhost` or a wrong IP). | Wait an hour; keep `SITE_ADDRESS` correct before retrying. |
| Containers fine but `https://…/api/health` refused from the internet | Firewall. Docker publishes ports through iptables and can bypass ufw. | `ufw status numbered` must show 443 allowed; `ss -tlnp` on the host should show only `docker-proxy` on 80/443 — if it lists 8000 or 11434, delete that `ports:` line from `deploy/docker-compose.yml`. |
| `docker compose up` says `env file deploy/.env not found` or builds with the wrong context | Compose was run from inside `deploy/`, so the project directory became `deploy/`. | Run `./deploy/deploy.sh` / `make`, or always `--project-directory /opt/tuk-convosearch`. |
| App restarts in a loop, logs show `ValueError: TUK_SECRET_KEY must be set...` | `TUK_ENVIRONMENT=production` with a short or missing secret. | `openssl rand -hex 32`, set `TUK_SECRET_KEY`, `dc up -d app`. |
| Log spam / disk full from logs | Unbounded json logs. | `bootstrap.sh` step 4 sets daemon-wide caps; `deploy/docker-compose.yml` sets them per service. Then `docker builder prune -f`. |
| Everything was fine, then answers got wrong after a redeploy | `data/` or `corpus/` were overwritten by a git/rsync from the laptop (`deploy/restore.sh` guards against this; a naive `rsync --delete` does not). | Restore a backup: `./deploy/restore.sh --latest`. Never rsync the repo up with `--delete` over `/opt/tuk-convosearch`. |
| `sqlite3` says `database disk image is malformed` | Someone copied `data/*.db` while the app was writing (the WAL sidecars were left behind). | Restore from a backup — `deploy/backup.sh` uses SQLite's online backup API precisely so this cannot happen. Never `cp` the live file. |

---

## 13. Reference: what lives where

| Path on the VPS | What it is | Backed up? |
|---|---|---|
| `/opt/tuk-convosearch` | the code you cloned/rsynced | no (git) |
| `/opt/tuk-convosearch/.env` | all configuration + the secret | **you, in a password manager** |
| `/opt/tuk-convosearch/data/` | `tuk_convosearch.db` — the whole store | yes (`backup.sh`) |
| `/opt/tuk-convosearch/corpus/tuk/` | source documents, incl. uploads | yes (`backup.sh`) |
| volume `tuk-ollama-models` | model weights | no — re-run `pull-models.sh` |
| volume `tuk-caddy-data` | TLS certs + ACME account | no — re-issued automatically |
| `/var/backups/tuk-convosearch` | last 14 nightly archives | rsync these off-box |
| `/var/log/tuk-backup.log` | cron output of the backup job | — |

Useful URLs once live: `https://ask.lemayian.com/` (students),
`https://ask.lemayian.com/admin`, `https://ask.lemayian.com/docs` (interactive API,
public by default — see the optional basic-auth block in `deploy/Caddyfile`),
`https://ask.lemayian.com/api/health` (readiness),
`https://ask.lemayian.com/api/config` (what the student UI reads at runtime).
