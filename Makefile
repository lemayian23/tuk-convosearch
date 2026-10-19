# TUK-ConvoSearch operator shortcuts.
#
# WHY HERE AND NOT deploy/Makefile: every target needs the repo root anyway —
# .env lives there, ./data and ./corpus live there, and `docker compose` must be
# run with --project-directory pointing at the repo root. Keeping the Makefile
# at the root means `make up` works from the directory you cloned into, which is
# the only thing the runbook has to remember.
#
#   make help          list everything
#   make env           create .env from deploy/.env.example with a fresh secret
#   make up            build (if needed) + start + wait for /api/health
#   make status        container states + readiness JSON
#   make logs          follow all service logs
#   make pull-models   download the models named in .env
#
# Requires: GNU make (default on Ubuntu), docker + compose plugin, and a repo
# that has been bootstrapped (deploy/bootstrap.sh). No target here is safe to
# run on your laptop against the production box; run them ON the VPS.

SHELL := /bin/bash

# Resolve the repo root from this Makefile's own location, so `make -C deploy up`
# and `make -f /opt/tuk-convosearch/Makefile up` both behave the same.
ROOT     := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
ROOT     := $(patsubst %/,%,$(ROOT))
COMPOSE  := deploy/docker-compose.yml
ENV_FILE := $(ROOT)/.env

# The three flags that make relative paths in the compose file mean "repo root"
# instead of "deploy/". See the header of deploy/docker-compose.yml.
DC := docker compose --project-directory $(ROOT) --env-file $(ENV_FILE) -f $(COMPOSE)

# The app container runs as uid 10001 (created in deploy/Dockerfile). Host-side
# data/corpus must be writable by it or indexing fails with a readonly-database
# error that has nothing to do with the code.
APP_UID := 10001

# SITE_ADDRESS is read straight out of .env (no `source`: the file may contain
# values with spaces, comments and shell metacharacters).
#
# NOTE FOR FUTURE EDITORS: never type a `#` inside a $(shell ...) call here —
# make strips it as a comment before the shell sees it, silently truncating the
# command. That is why there is no sed comment-stripper below: `firstword` takes
# the value and drops anything after it, including an inline comment. Quotes are
# translated to spaces for the same reason.
SITE_ADDRESS := $(firstword $(shell grep -E '^[[:space:]]*(export[[:space:]]+)?SITE_ADDRESS[[:space:]]*=' $(ENV_FILE) 2>/dev/null | tail -n 1 | cut -d= -f2- | tr '"' ' ' | tr "'" ' '))
SITE_ADDRESS := $(if $(SITE_ADDRESS),$(SITE_ADDRESS),ask.lemayian.com)

.DEFAULT_GOAL := help
.PHONY: help env up down restart rebuild logs status ps backup restore pull-models \
        bench shell ollama-shell sqlite reindex admin health mkdirs clean-rollback

help: ## Show this help
	@printf '\nTUK-ConvoSearch — %s\n\n' $(SITE_ADDRESS)
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'
	@printf '\n  First time on a new VPS: deploy/RUNBOOK.md (start at step 1).\n\n'

# --------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------- #
env: ## Create .env from deploy/.env.example with a freshly generated secret
	@test -f $(ENV_FILE) && printf '\033[33m.env already exists; not touching it.\033[0m\n' || \
		cp deploy/.env.example $(ENV_FILE)
	@test -f $(ENV_FILE) || (printf '\033[31mmissing deploy/.env.example\033[0m\n'; exit 1)
	# Replace the placeholder with `openssl rand -hex 32`. Done with a temp file
	# because sed -i on a real .env would otherwise expose it world-readable for
	# a moment; umask 077 + mv keeps it 600 throughout.
	@umask 077; \
		secret="$$([ -f $(ENV_FILE) ] && sed -n 's/^TUK_SECRET_KEY=\(.*\)$$/\1/p' $(ENV_FILE) | head -n1)"; \
		if [ "$$secret" = "REPLACE_ME_WITH_OPENSSL_RAND_HEX_32" ] || [ -z "$$secret" ]; then \
			new=$$(openssl rand -hex 32); \
			sed "s|^TUK_SECRET_KEY=.*|TUK_SECRET_KEY=$$new|" $(ENV_FILE) > $(ENV_FILE).tmp && \
			mv $(ENV_FILE).tmp $(ENV_FILE) && chmod 600 $(ENV_FILE); \
			printf 'TUK_SECRET_KEY set (64 hex chars, length checked by deploy.sh).\n'; \
		else \
			printf 'TUK_SECRET_KEY already populated (%s chars) - left alone.\n' "$${#secret}"; \
		fi
	@printf '\nNow edit $(ENV_FILE):\n'
	@printf '  SITE_ADDRESS, TUK_BOOTSTRAP_ADMIN_EMAIL / _PASSWORD (or leave blank\n'
	@printf '  and use: make admin). Everything else has a working default.\n'
	@printf '  chmod 600 .env && git check-ignore .env\n\n'

mkdirs: ## Create data/ and corpus/tuk/ with the right owner for uid 10001
	@mkdir -p $(ROOT)/data $(ROOT)/corpus/tuk
	@if [ "$$(id -u)" = "0" ]; then \
		chown -R $(APP_UID):$(APP_UID) $(ROOT)/data $(ROOT)/corpus; \
		printf 'data/ and corpus/ owned by $(APP_UID).\n'; \
	else \
		sudo chown -R $(APP_UID):$(APP_UID) $(ROOT)/data $(ROOT)/corpus 2>/dev/null \
			|| printf '\033[33mCould not chown to $(APP_UID). If the app cannot write, run:\n  sudo chown -R $(APP_UID):$(APP_UID) $(ROOT)/data $(ROOT)/corpus\033[0m\n'; \
	fi

# --------------------------------------------------------------------- #
# Lifecycle
# --------------------------------------------------------------------- #
up: mkdirs ## Build if needed, start, and wait until /api/health is ready
	@deploy/deploy.sh

rebuild: mkdirs ## Force a clean image build (slow; after editing requirements.txt)
	@deploy/deploy.sh --rebuild

down: ## Stop and remove containers; volumes, images, data and certs all survive
	@$(DC) down
	@printf 'Containers removed. Named volumes (tuk-ollama-models, tuk-caddy-data) are intact.\n'

restart: ## Restart the app container only (no rebuild); ~1-2 min to become ready
	@$(DC) restart app
	@printf 'Waiting for readiness... '
	@for i in $$(seq 1 24); do \
		if $(DC) exec -T app curl -fsS http://127.0.0.1:8000/health >/dev/null 2>&1; then printf ' alive.\n'; break; fi; \
		sleep 5; \
	done
	@$(DC) exec -T app python -c 'import json,urllib.request; d=json.load(urllib.request.urlopen("http://127.0.0.1:8000/api/health",timeout=10)); print(" status=",d["status"],"checks=",d["checks"])' \
		|| printf '\033[33m app still starting; run: make status\033[0m\n'

logs: ## Follow logs from every service (Ctrl-C to stop)
	@$(DC) logs -f --tail=100

status: ## Container states, then the readiness payload
	@$(DC) ps
	@printf '\n/api/health:\n'
	@$(DC) exec -T app python -c 'import json,urllib.request; print(json.dumps(json.load(urllib.request.urlopen("http://127.0.0.1:8000/api/health",timeout=10)),indent=2))' \
		|| printf '\033[33mThe app is not answering. make logs\033[0m\n'

ps: ## Short version of status (container table only)
	@$(DC) ps

health: ## One-line readiness check, suitable for monitoring
	@$(DC) exec -T app python -c 'import json,urllib.request; d=json.load(urllib.request.urlopen("http://127.0.0.1:8000/api/health",timeout=10)); print(d["status"], d["checks"])'

# --------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------- #
pull-models: ## Pull the models named in .env (or: make pull-models MODEL=qwen2.5:7b-instruct)
	@if [ -n "$(MODEL)" ]; then deploy/pull-models.sh "$(MODEL)"; else deploy/pull-models.sh; fi

bench: ## Time llama3.2:1b vs 3b vs qwen2.5:7b on this box, print the decision table
	@deploy/pull-models.sh --bench

# --------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------- #
backup: ## Snapshot the database + corpus into /var/backups/tuk-convosearch
	@deploy/backup.sh

restore: ## Restore an archive: make restore ARCHIVE=/var/backups/tuk-convosearch/tuk-backup-XXXX.tar.gz
	@if [ -z "$(ARCHIVE)" ]; then \
		deploy/restore.sh --list; \
		printf '\nNow run: make restore ARCHIVE=<one of the above>\n'; \
	else \
		deploy/restore.sh "$(ARCHIVE)"; \
	fi

reindex: ## Re-ingest every file in the corpus (no login; runs inside the container)
	@printf 'Re-reading the corpus inside the app container...\n'
	@printf '  Files already embedded are skipped by content hash (status=unchanged),\n'
	@printf '  so this is safe to run on a full corpus. It is synchronous: a large\n'
	@printf '  corpus embeds on CPU for several minutes before the prompt returns.\n'
	@$(DC) exec -T app python -c 'from app.config import get_settings; from app.ingestion.documents import SUPPORTED_EXTENSIONS; from app.ingestion.index import get_index; s = get_settings(); ix = get_index(); files = sorted(p for p in s.corpus_dir.glob("*") if p.suffix.lower() in SUPPORTED_EXTENSIONS); [print(r["status"], p.name, r.get("chunks", 0)) for p, r in ((p, ix.ingest_path(p)) for p in files)]; print("corpus_dir:", s.corpus_dir, "files:", len(files)); print("index:", ix.get_stats())'
	@printf '\nDone. Verify: make health\n'

# --------------------------------------------------------------------- #
# Shell access
# --------------------------------------------------------------------- #
shell: ## Interactive bash inside the app container (as uid 10001)
	@$(DC) exec app bash

ollama-shell: ## Interactive bash inside the ollama container
	@$(DC) exec ollama bash

admin: ## Create the first admin account (email from: make admin EMAIL=you@tuk.ac.ke)
	@if [ -z "$(EMAIL)" ]; then \
		printf 'Usage: make admin EMAIL=you@tuk.ac.ke\n'; \
	else \
		$(DC) exec app python scripts/create_admin.py --email "$(EMAIL)"; \
	fi

# The psql-equivalent for this project. SQLite is the whole store, so there is
# no server to talk to: opening the file IS the client.
#   - mode=ro  : reading with the write lock would let a stray UPDATE corrupt a
#                production file, and it is also why we never inspect the live db
#                with a plain `sqlite3 data/...`.
#   - immutable? no: WAL readers need normal readonly mode, which replays the WAL.
# Needs the sqlite3 CLI (installed by deploy/bootstrap.sh). If the file is not
# world-readable, use `make shell` and python's sqlite3 module instead.
sqlite: ## Open the live database read-only (psql equivalent for this stack)
	@command -v sqlite3 >/dev/null || { printf 'sqlite3 CLI missing: sudo apt-get install -y sqlite3\n'; exit 1; }
	@test -f $(ROOT)/data/tuk_convosearch.db || { printf 'No database yet. Start the stack first (make up).\n'; exit 1; }
	@DB='file:$(ROOT)/data/tuk_convosearch.db?mode=ro'; \
	 sqlite3 -readonly "$$DB" "SELECT 'documents = ' || COUNT(*) FROM documents" ; \
	 sqlite3 -readonly "$$DB" "SELECT 'chunks      = ' || COUNT(*) FROM chunks" ; \
	 sqlite3 -readonly "$$DB" "SELECT 'sessions    = ' || COUNT(*) FROM sessions" ; \
	 sqlite3 -readonly "$$DB" "SELECT 'queries     = ' || COUNT(*) FROM query_log" ; \
	 sqlite3 -readonly "$$DB" ".tables" ; \
	 printf '\nMost recent student questions:\n' ; \
	 sqlite3 -readonly "$$DB" -header -column "SELECT substr(question,1,48) AS question, chunks_found, round(response_time,1) AS secs, datetime(timestamp,'unixepoch') AS utc FROM query_log ORDER BY timestamp DESC LIMIT 12"
	@printf '\nFor an interactive session (safe: read-only):\n  sqlite3 -readonly "file:$(ROOT)/data/tuk_convosearch.db?mode=ro"\n'
	@printf 'Then: .schema documents | .headers on | SELECT * FROM chunks LIMIT 1;\n'

clean-rollback: ## Delete the rollback image tag (frees a few hundred MB)
	@docker image rm tuk-convosearch-app:rollback 2>/dev/null && printf 'Removed tuk-convosearch-app:rollback\n' || printf 'Nothing to remove.\n'
