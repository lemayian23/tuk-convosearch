#!/usr/bin/env bash
# =====================================================================
# deploy/pull-models.sh — download the models this stack is configured to use.
#
#   ./deploy/pull-models.sh              pull TUK_OLLAMA_MODEL + reasoning model
#   ./deploy/pull-models.sh --list       what is already on the box
#   ./deploy/pull-models.sh --bench      time 1b vs 3b vs 7b + a real prefill run
#   ./deploy/pull-models.sh llama3.2:3b  pull one specific tag
#
# Why this exists as a script and not just `ollama pull`:
#   1. The Ollama container has no published port, so you cannot reach it from
#      your laptop, and `ollama` on the host would talk to nothing. The only
#      correct invocation is *inside* the container, where OLLAMA_HOST resolves
#      to the daemon's own socket.
#   2. The model names live in .env. Repeating them on a command line is how a
#      deploy ends up with llama3.2:1b pulled but llama3.2:3b configured, and a
#      permanently "degraded" /api/health.
#   3. Model weights are NOT backed up (see deploy/backup.sh) — after a volume
#      wipe or a new VPS this script is the recovery step.
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

compose() {
	docker compose --project-directory "$REPO_ROOT" --env-file "$ENV_FILE" -f "$COMPOSE_FILE" "$@"
}

# ------------------------------------------------------------------ #
# Read a key out of .env without assuming it is exported into this shell.
# Deliberately NOT `source .env`: the file can contain values with spaces,
# inline comments and shell metacharacters, and sourcing would also drag the
# secret key into this process's environment.
#   - tolerates `KEY=value`, `export KEY=value`, spaces around `=`
#   - last assignment wins (matches what compose does)
#   - strips a trailing comment only when whitespace precedes the '#'
#   - strips one layer of matching quotes
#   - returns empty (not an error) when the key is absent or commented out
# Known limitation: a *quoted* value that itself contains " #" is truncated,
# because the comment is stripped before the quotes. Model names, hosts and
# critic modes never contain that; if you ever need it, use deploy/deploy.sh's
# simpler reader or keep the value unquoted without spaces.
# ------------------------------------------------------------------ #
read_env() {
	local key="$1" default="${2:-}" line
	[[ -f "$ENV_FILE" ]] || { printf '%s' "$default"; return 0; }
	line="$(grep -E "^[[:space:]]*(export[[:space:]]+)?${key}[[:space:]]*=" "$ENV_FILE" | tail -n 1 || true)"
	if [[ -z "$line" ]]; then printf '%s' "$default"; return 0; fi
	line="${line#*=}"                                     # everything after first '='
	line="${line#"${line%%[![:space:]]*}"}"               # ltrim
	if [[ "$line" =~ ([[:space:]]+)# ]]; then            # trailing comment, needs whitespace before #
		line="${line%%[[:space:]]#*}"
	fi
	line="${line%"${line##*[![:space:]]}"}"              # rtrim
	[[ "${line:0:1}" == '"' && "${line: -1}" == '"' ]] && line="${line:1:-1}"
	[[ "${line:0:1}" == "'" && "${line: -1}" == "'" ]] && line="${line:1:-1}"
	printf '%s' "${line:-$default}"
}

MODEL="$(read_env TUK_OLLAMA_MODEL llama3.2:1b)"
REASONING="$(read_env TUK_OLLAMA_REASONING_MODEL llama3.2:3b)"
HOST="$(read_env TUK_OLLAMA_HOST http://ollama:11434)"
KEEP_ALIVE="$(read_env TUK_LLM_KEEP_ALIVE 30m)"
CONCURRENCY="$(read_env TUK_MAX_CONCURRENT_LLM_CALLS 2)"

# ------------------------------------------------------------------ #
# Guard: the ollama container must be up. `docker compose run` would also work
# but the pull is faster and less confusing if the long-lived container already
# owns the volume.
# ------------------------------------------------------------------ #
require_ollama() {
	if ! compose ps -q ollama | grep -q .; then
		say "The ollama service is not created yet; starting just that container"
		compose up -d ollama
	fi
	# Wait for the API rather than sleeping a fixed number of seconds.
	local i
	for i in $(seq 1 30); do
		if compose exec -T ollama ollama list >/dev/null 2>&1; then return 0; fi
		sleep 2
	done
	die "Ollama is not answering inside its container. Check: ./deploy/deploy.sh --logs"
}

# True when the tag is already in `ollama list`.
# Note the two accepted spellings: `ollama list` prints the manifest name, and
# depending on the Ollama version `llama3.2:1b` shows up as "llama3.2:1b" or as
# "llama3.2:1b:latest". Matching both avoids re-downloading a model that is
# already there; matching whole lines (grep -x) avoids "llama3.2:1b" counting as
# present when only "llama3.2:1b-instruct-q4_K_M" is installed.
has_model() {
	compose exec -T ollama ollama list 2>/dev/null | awk '{print $1}' |
		grep -qxF -e "$1" -e "$1:latest"
}

# ------------------------------------------------------------------ #
# Decision table: printed for the same reason the RAM table is in .env.example
# — the choice of model is the single biggest quality/latency decision in this
# deployment, and it depends on hardware nobody can guess for you.
# ------------------------------------------------------------------ #
decision_table() {
	cat <<'TABLE'
  On CPU, with one model slot per 4 GB free RAM. Rough numbers for a 4 vCPU /
  8 GB Kenya VPS; measure with --bench before you trust them.

  +-----------------+---------+-----------+-----------------+------------------+
  | model           | disk    | RAM while | tokens/sec (CPU)| answer latency   |
  |                 |         | serving   |                 | for a 150-token  |
  |                 |         |           |                 | grounded answer  |
  +-----------------+---------+-----------+-----------------+------------------+
  | llama3.2:1b     | ~1.3 GB | ~2.0 GB   | 15-25 tok/s     | 7-12 s           |
  | llama3.2:3b     | ~2.0 GB | ~3.5 GB   |  6-12 tok/s     | 15-25 s          |
  | qwen2.5:7b-inst | ~4.7 GB | ~7.5 GB   |  2-5 tok/s      | 40-90 s          |
  +-----------------+---------+-----------+-----------------+------------------+

  Quality vs cost, for THIS corpus (short factual questions: dates, fees,
  registration rules, where to submit a form):

    llama3.2:1b  - correct answer ~8/10 when the chunk is clean; noticeably
                   worse on multi-hop questions ("after rescheduling, which
                   group sits on Tuesday?"). The safe default on 4 GB.
    llama3.2:3b  - ~9/10, far fewer fluent-but-wrong numbers. Needs 8 GB and
                   TUK_MAX_CONCURRENT_LLM_CALLS=2. Best quality/latency point
                   for a single-box student deployment.
    qwen2.5:7b   - ~9.5/10 and the best KisEnglish instruction following, but
                   on CPU one question takes a minute. Only worth it on 16 GB
                   with low concurrency (set TUK_MAX_CONCURRENT_LLM_CALLS=2),
                   and only if you accept students waiting.

  Remember the retrieval does the grounding work: a wrong date in the answer is
  usually a missing document in corpus/, not a small model. Fix the corpus first.
TABLE
}

# ------------------------------------------------------------------ #
# Modes
# ------------------------------------------------------------------ #
MODE="pull"
TARGET=""
for arg in "$@"; do
	case "$arg" in
		--list)   MODE="list" ;;
		--bench)  MODE="bench" ;;
		--table)  decision_table; exit 0 ;;
		-h|--help) sed -n '2,20p' "$0"; exit 0 ;;
		-*)       die "Unknown option: $arg (try --list, --bench, --table)" ;;
		*)        TARGET="$arg" ;;
	esac
done

command -v docker >/dev/null 2>&1 || die "docker not installed. Run deploy/bootstrap.sh as root."
[[ -f "$COMPOSE_FILE" ]] || die "Missing $COMPOSE_FILE"
[[ -f "$ENV_FILE" ]] || die "No $ENV_FILE at the repo root. Copy deploy/.env.example to .env first."
cd "$REPO_ROOT"

if [[ "$MODE" == "list" ]]; then
	require_ollama
	say "Models currently on this box"
	compose exec -T ollama ollama list
	say "Configured in .env"
	printf '    TUK_OLLAMA_MODEL           = %s\n' "$MODEL"
	printf '    TUK_OLLAMA_REASONING_MODEL = %s\n' "$REASONING"
	if has_model "$MODEL"; then ok "the configured model is present"; else
		warn "MISSING: $MODEL — /api/health reports degraded until you pull it."
		printf '    ./deploy/pull-models.sh\n'
	fi
	if [[ "$REASONING" != "$MODEL" ]]; then
		if has_model "$REASONING"; then ok "reasoning model present"; else
			warn "reasoning model $REASONING not present (only needed if TUK_CRITIC_MODE=llm)"
		fi
	fi
	exit 0
fi

# ------------------------------------------------------------------ #
# Prefill bench: the number the loop below does NOT measure.
#
# `ollama run "question"` sends ~20 tokens. A real answer is generated after
# the retrieved context, and on CPU that context is evaluated token by token
# before the first word appears. Measured cold on a laptop, llama3.2:1b, model
# unloaded between trials and a random prefix so Ollama's KV reuse cannot help:
# 924 prompt tokens = 41.5s of prefill, 432 = 20.0s. That is the wait a student
# sees before anything appears, and TUK_LLM_TIMEOUT_SECONDS has to cover it plus
# the whole generated answer.
#
# Runs in the app image so the prompt is built by the same pipeline the server
# uses, with the same num_ctx/num_predict/keep_alive it will send.
# ------------------------------------------------------------------ #
prefill_bench() {
	say "Prefill: what one grounded answer actually costs this box"
	warn "If the app image is not built yet this is skipped - run ./deploy/deploy.sh first."
	# The Python below prints its own aligned report; bash only relays it, so the
	# numbers cannot drift out of step with the labels.
	compose run --rm -T app python - <<'PY' || warn "Could not run the bench in the app container (image missing or ollama unreachable)"
import json, math, random, urllib.request

from app.config import get_settings
from app.rag import RAGContext
from app.rag.configs.tuk_pipeline import TUK_SYSTEM_PROMPT

QUESTION = "When do the second semester exams start?"
s = get_settings()

context, origin = "", "retrieved from the indexed corpus"
try:
    from app.services.pipelines import get_pipeline

    pipeline = get_pipeline("tuk")
    ctx = RAGContext(question=QUESTION, session_id="bench", top_k=s.top_k)
    for agent in pipeline.agents[:2]:
        agent.run(ctx)
    context = ctx.metadata.get("context_string") or ""
    origin = "retrieved context: %d chunks kept of %d retrieved" % (
        len(ctx.retrieved_chunks), ctx.top_k)
except Exception as exc:
    origin = "no retrieved context (%s)" % type(exc).__name__

if not context.strip():
    origin += " - nothing indexed, using a synthetic ~430-token prompt (run: make reindex)"
    filler = (
        "2026 Academic Year Calendar: Semester II examinations begin in May, "
        "Continuous Assessment Tests run in March, and industrial attachment "
        "covers twelve weeks after the final exam week. "
    )
    while len(context) < 1700:
        context += filler

# Random prefix: Ollama reuses the KV cache for any prompt that starts the same
# way, and a bench that runs twice would otherwise report the cache, not the box.
prompt = TUK_SYSTEM_PROMPT.format(
    context="Bench run %04x. " % random.randrange(1 << 16) + context,
    history="No previous conversation.",
    question=QUESTION)
body = {
    "model": s.ollama_model,
    "prompt": prompt,
    "stream": False,
    "keep_alive": s.llm_keep_alive,
    "options": {
        "num_predict": s.synthesis_num_predict,
        "temperature": s.synthesis_temperature,
        "num_ctx": s.synthesis_num_ctx,
        "repeat_penalty": s.synthesis_repeat_penalty,
    },
}
with urllib.request.urlopen(
    urllib.request.Request(
        s.ollama_host.rstrip("/") + "/api/generate",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    ),
    timeout=900,
) as response:
    r = json.load(response)

prefill = r.get("prompt_eval_duration", 0) / 1e9
gen_time = r.get("eval_duration", 0) / 1e9
gen_tokens = r.get("eval_count", 0) or 1
prefill_tps = (r.get("prompt_eval_count", 0) / prefill) if prefill else 0.0
gen_tps = (gen_tokens / gen_time) if gen_time else 0.0
# A full-length answer writes num_predict tokens; the measured one may be shorter.
worst = prefill + (s.synthesis_num_predict / gen_tps if gen_tps else 0.0)
suggested = max(60, int(math.ceil(worst * 2 / 15.0)) * 15)

print('    model          %s  (weights loaded in %.1f s here; the app does that at boot)'
      % (s.ollama_model, r.get("load_duration", 0) / 1e9))
print('    prompt tokens  %s (context: %s)' % (r.get("prompt_eval_count"), origin))
print('    prefill        %.1f s  (%.1f tok/s)  <- nothing on screen yet' % (prefill, prefill_tps))
print('    generation     %.1f s  (%.1f tok/s)  %d tokens written' % (gen_time, gen_tps, gen_tokens))
print('    this answer    %.1f s wall' % (prefill + gen_time))
print('    longest answer %.0f s  (a full %d-token reply at %.1f tok/s)'
      % (worst, s.synthesis_num_predict, gen_tps))
print()
if worst > s.llm_timeout_seconds:
    print('    !! TUK_LLM_TIMEOUT_SECONDS=%g is BELOW that. A long answer gets cut'
          % s.llm_timeout_seconds)
    print('       off mid-sentence with a 504. Raise it to %d, or shrink the prompt'
          % suggested)
    print('       with TUK_MAX_CHUNKS_PER_SOURCE / TUK_MIN_RELEVANCE_MARGIN.')
else:
    print('    TUK_LLM_TIMEOUT_SECONDS=%g covers the longest answer with %.1fx to spare.'
          % (s.llm_timeout_seconds, s.llm_timeout_seconds / worst if worst else 0.0))
    print('    If this box is slower than your VPS, re-run this bench there rather')
    print('    than guessing: prefill is what moves, and it is all prompt length.')
PY
}

if [[ "$MODE" == "bench" ]]; then
	require_ollama
	say "Hardware"
	printf '    CPU cores : %s\n' "$(nproc)"
	free -h | awk '/^Mem:/ {print "    RAM       : " $2 "\n    available : " $7}'
	printf '    Swap      : %s\n' "$(free -h | awk '/^Swap:/ {print $2}')"
	printf '    Disk free : %s\n\n' "$(df -h / | awk 'NR==2 {print $4}')"
	decision_table
	echo
	say "Benchmark: the same real question through each model"
	warn "Each line is one cold-ish generation of a BARE question - no retrieved"
	warn "context, so these understate real answer time by the whole prefill. The"
	warn "prefill bench after them is the number students actually wait for."
	warn "Total wall time scales with the number of models you pull AND keep"
	warn "resident; Ollama evicts to disk."
	printf '\n    Question: "When do the second semester exams start?"\n\n'

	for m in llama3.2:1b llama3.2:3b qwen2.5:7b-instruct; do
		if ! has_model "$m"; then
			printf '    %-22s not installed -> ./deploy/pull-models.sh %s\n' "$m" "$m"
			continue
		fi
		START=$(date +%s)
		# keep_alive=0 so the model is unloaded afterwards and the next timing
		# starts on equal footing (no RAM contention between trials).
		if OUT="$(compose exec -T ollama ollama run "$m" "When do the second semester exams start? Answer in one sentence." --keepalive 0 2>/dev/null | tr -d '\n' | cut -c1-90)"; then
			END=$(date +%s)
			printf '    %-22s %4d s   %s\n' "$m" "$((END - START))" "$OUT"
		else
			printf '    %-22s FAILED (out of memory? try 30s then retry)\n' "$m"
		fi
	done

	echo
	prefill_bench
	echo
	warn "Set your winner in .env (TUK_OLLAMA_MODEL=...) and re-run this script"
	warn "with no arguments, then ./deploy/deploy.sh --status to confirm"
	warn "checks.model_present=true. Do not edit .env to a model you have not"
	warn "pulled: the app will boot, warm_model will fail, and health stays degraded."
	exit 0
fi

# ------------------------------------------------------------------ #
# Default mode: pull what .env says we need
# ------------------------------------------------------------------ #
require_ollama

declare -a WANTED=()
if [[ -n "$TARGET" ]]; then
	WANTED=("$TARGET")
else
	WANTED=("$MODEL")
	# The reasoning model is only worth several GB if the critic actually calls
	# it. TUK_CRITIC_MODE=heuristic never does.
	if [[ "$(read_env TUK_CRITIC_MODE heuristic)" == "llm" && "$REASONING" != "$MODEL" ]]; then
		WANTED+=("$REASONING")
	else
		say "Skipping TUK_OLLAMA_REASONING_MODEL ($REASONING)"
		printf '    TUK_CRITIC_MODE is heuristic, so nothing calls the reasoning model.\n'
		printf '    Set TUK_CRITIC_MODE=llm in .env if you want the stronger grounding check.\n'
	fi
fi

say "Models this deployment needs"
for m in "${WANTED[@]}"; do printf '    %s\n' "$m"; done
printf '    Ollama API used: %s (inside the container; never published on the host)\n' "$HOST"
echo
warn "DOWNLOAD SIZE WARNING: these are multi-gigabyte downloads over the data"
warn "centre link, and your first question after each pull also pays a load cost."
warn "On a metered or slow connection, expect 5-30 minutes per model. The models"
warn "themselves are NOT included in deploy/backup.sh — re-pull after a rebuild."
echo

for m in "${WANTED[@]}"; do
	if has_model "$m"; then
		ok "$m already present (nothing to download)"
		continue
	fi
	say "Pulling $m"
	# No pipe/redirect: keep the live progress bar Ollama prints, because a
	# silent multi-GB download looks exactly like a hung script.
	compose exec -T ollama ollama pull "$m" || die "Failed to pull $m. If the download broke mid-way, just re-run this script; Ollama resumes from existing layers."
done

say "Verifying against Ollama's own /api/tags"
# This is the exact endpoint the app's model_status() hits (app/services/ollama.py),
# so if this passes, checks.model_present in /api/health will pass too.
TAGS_JSON="$(compose exec -T app python -c '
import json, urllib.request, os
host = os.environ.get("TUK_OLLAMA_HOST", "http://ollama:11434").rstrip("/")
print(json.dumps([m.get("name","") for m in json.load(urllib.request.urlopen(host + "/api/tags", timeout=10)).get("models", [])]))
' 2>/dev/null)" || die "Could not reach $HOST from the app container. That is the real bug - check TUK_OLLAMA_HOST (now: $HOST)."
printf '    models visible to the app: %s\n' "$TAGS_JSON"

for m in "${WANTED[@]}"; do
	if printf '%s' "$TAGS_JSON" | grep -qE "\"${m}(:latest)?\""; then
		ok "$m visible to the app on /api/tags"
	else
		die "$m is not in /api/tags. Pull reported success but the volume may be wrong; ./deploy/deploy.sh --logs"
	fi
done

say "Warm-up (optional)"
printf 'The app warms %s at boot when TUK_WARM_MODEL_ON_BOOT=true, so the first\nstudent does not pay the cold load. To do it now instead of restarting the app:\n\n' "$MODEL"
printf '    docker compose --project-directory %s --env-file %s -f %s exec app \\\n' "$REPO_ROOT" "$ENV_FILE" "$COMPOSE_FILE"
printf '      python -c "from app.services.ollama import warm_model; from app.config import get_settings as g; print(warm_model(g().ollama_model, g().llm_keep_alive))"\n\n'
printf 'Then confirm end to end:\n\n'
printf '    ./deploy/deploy.sh --status\n'
ok "done"
