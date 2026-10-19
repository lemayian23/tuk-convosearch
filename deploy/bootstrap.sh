#!/usr/bin/env bash
# =====================================================================
# deploy/bootstrap.sh — first-time setup for a fresh Ubuntu 24.04 LTS VPS.
#
# Run it ONCE, as root, before anything else:
#     sudo bash deploy/bootstrap.sh
#
# It is idempotent: every step checks whether it has already been done, so
# re-running it after a mistake is safe and boring. It does NOT deploy the app;
# that is deploy/deploy.sh after you have copied the repo and created .env.
#
# WHAT IT DOES   (1) apt update + base packages
#                (2) Docker Engine + Compose plugin from Docker's own repo
#                (3) swapfile + vm.swappiness (model loading needs it)
#                (4) Docker daemon log-size caps in /etc/docker/daemon.json
#                (5) ufw: allow 22/80/443 only — explicitly NOT 11434 or 8000
#                (6) fail2ban with a sane sshd jail
#                (7) a non-root `deploy` user in the docker group
#                (8) the /opt/tuk-convosearch directory and a nightly backup cron
#
# WHAT IT DELIBERATELY DOES NOT DO
#   * It does not disable SSH password authentication, and does not touch
#     sshd_config beyond fail2ban's `maxstart`-style settings. Password-off is
#     step (9) at the very end and it prints the exact commands, because doing
#     it before your key works is how people lock themselves out of a VPS.
#     Keep this terminal OPEN while you test the key.
#   * It does not create .env, pull models, or start containers.
#   * It does not reboot for you (it tells you when to).
#
# Docker install method — why the apt repo and not `get.docker.com`:
#   The official convenience script is one long pipe-to-bash that installs a
#   .deb repo anyway. Adding the repo explicitly with a pinned GPG key is the
#   same end state, is auditable, survives re-runs, and gives you normal
#   `apt upgrade docker-ce` updates. Ubuntu 24.04 also ships docker.io in the
#   main archive, but its Compose plugin lags badly and the containerd/overlay
#   defaults differ from what the app images are built against, so we use
#   Docker's own repo for both.
# =====================================================================
set -euo pipefail

# ------------------------------------------------------------------ #
# Small helpers
# ------------------------------------------------------------------ #
BLUE='\033[1;34m'; GREEN='\033[1;32m'; YELLOW='\033[1;33m'; RED='\033[1;31m'; OFF='\033[0m'
say()  { printf '\n%b==> %s%b\n' "$BLUE" "$*" "$OFF"; }
ok()   { printf '%b    done: %s%b\n' "$GREEN" "$*" "$OFF"; }
skip() { printf '%b    already: %s%b\n' "$YELLOW" "$*" "$OFF"; }
die()  { printf '%b    FAILED: %s%b\n' "$RED" "$*" "$OFF" >&2; exit 1; }

APP_DIR="${APP_DIR:-/opt/tuk-convosearch}"
DEPLOY_USER="${DEPLOY_USER:-deploy}"
SWAP_MB="${SWAP_MB:-4096}"

# Absolute path of the directory holding this script, so the runbook can call
# it from anywhere (`sudo bash /opt/tuk-convosearch/deploy/bootstrap.sh`).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

# Refuse to run somewhere that is not Ubuntu 24.04. `apt-get` flags and the
# docker repo codename below assume it; running this on Debian 12 or 22.04
# would write a broken repo file.
if [[ -r /etc/os-release ]]; then
	# shellcheck disable=SC1091
	. /etc/os-release
	if [[ "${ID:-}" != "ubuntu" || "${VERSION_ID:-}" != "24.04" ]]; then
		die "This script targets Ubuntu 24.04 LTS. You are on ${ID:-unknown} ${VERSION_ID:-?}. Edit it or run the steps by hand."
	fi
else
	die "/etc/os-release not found; cannot confirm the distribution."
fi

if [[ $EUID -ne 0 ]]; then
	die "Run as root:  sudo bash deploy/bootstrap.sh"
fi

say "Target: Ubuntu ${VERSION_ID} (${UBUNTU_CODENAME:-noble}) — app dir $APP_DIR, deploy user $DEPLOY_USER"
if [[ "${1:-}" == "--dry-run" ]]; then
	echo "Dry run: nothing will be changed. Re-run without --dry-run to apply."
	exit 0
fi

# Confirm before anything that could interrupt a running service.
if [[ -x /usr/bin/docker || -d "$APP_DIR" ]]; then
	say "This machine already has Docker and/or $APP_DIR — a previous run, or an app already deployed."
	printf "Continue anyway? Every step is checked individually, but re-running changes system config. [y/N] "
	read -r reply
	[[ "${reply:-n}" =~ ^[Yy]$ ]] || die "Aborted by operator."
fi

# ------------------------------------------------------------------ #
# 1. Base packages
# ------------------------------------------------------------------ #
say "1/8 Updating apt index and installing base packages"
export DEBIAN_FRONTEND=noninteractive
apt-get update -y
# ca-certificates/curl/gnupg: needed for the Docker repo.
# ufw/fail2ban: the firewall and the sshd jail.
# sqlite3: reading/verifying the backup (.backup / VACUUM INTO) and ad-hoc queries.
# rsync: copying the repo up and pulling backups down.
# unattended-upgrades: security patches without you logging in.
# swapfile-utils is not a package; falloc comes from util-linux (already installed).
apt-get install -y --no-install-recommends \
	ca-certificates curl gnupg xz-utils \
	ufw fail2ban \
	sqlite3 rsync unzip \
	unattended-upgrades apt-listchanges \
	bash-completion
ok "base packages present"

# ------------------------------------------------------------------ #
# 2. Docker Engine + Compose plugin (Docker's own apt repo)
# ------------------------------------------------------------------ #
if command -v docker >/dev/null 2>&1 && docker compose version >/dev/null 2>&1; then
	skip "docker $(docker --version | awk '{print $3}' | tr -d ',') + compose plugin"
else
	say "2/8 Installing Docker Engine + Compose plugin from download.docker.com"
	install -m 0755 -d /etc/apt/keyrings
	if [[ ! -f /etc/apt/keyrings/docker.gpg ]]; then
		curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
			| gpg --dearmor -o /etc/apt/keyrings/docker.gpg
		chmod a+r /etc/apt/keyrings/docker.gpg
	fi
	# The `stable` channel, not `nightly`. Pinning the codename keeps `apt update`
	# honest if you later move the machine to another release.
	echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
https://download.docker.com/linux/ubuntu ${UBUNTU_CODENAME:-noble} stable" \
		> /etc/apt/sources.list.d/docker.list
	apt-get update -y
	apt-get install -y --no-install-recommends \
		docker-ce docker-ce-cli containerd.io \
		docker-buildx-plugin docker-compose-plugin
	ok "docker + compose plugin installed"
fi

systemctl enable --now docker
docker compose version
# Sanity check that the daemon actually answers, before you discover it later.
if [[ "$(docker info --format '{{.ServerVersion}}' 2>/dev/null || echo none)" == "none" ]]; then
	die "docker daemon is not responding. Check: systemctl status docker --no-pager"
fi

# ------------------------------------------------------------------ #
# 3. Swap — non-negotiable on a small VPS running an LLM
# ------------------------------------------------------------------ #
# Why: a 4 GB box loading llama3.2:3b plus the Python embedder can exceed
# physical RAM for a few seconds. Without swap the OOM killer takes down
# whichever container it likes; with swap it just gets slow once. This is a
# CPU-only deployment, so the model files are read into RAM by Ollama directly.
say "3/8 Swap"
if [[ "$(free -m | awk '/^Swap:/ {print $2}')" -gt 512 ]]; then
	skip "swap already active ($(free -h | awk '/^Swap:/ {print $2}'))"
else
	if [[ -f /swapfile ]]; then
		skip "/swapfile exists but is not swapon; check 'swapon --show'"
	else
		printf "Create a %s MB swapfile at /swapfile now? (a few seconds; needs disk space) [Y/n] " "$SWAP_MB"
		read -r reply
		if [[ ! "${reply:-y}" =~ ^[Nn]$ ]]; then
			# fallocate is instant; fall back to dd on filesystems without it.
			if ! fallocate -l "${SWAP_MB}M" /swapfile 2>/dev/null; then
				dd if=/dev/zero of=/swapfile bs=1M count="$SWAP_MB" status=none
			fi
			chmod 600 /swapfile
			mkswap /swapfile >/dev/null
			swapon /swapfile
			# Persist across reboot. Guard against a duplicate fstab line on re-run.
			grep -q '^/swapfile' /etc/fstab || echo '/swapfile none swap sw 0 0' >> /etc/fstab
			ok "${SWAP_MB} MB swapfile online"
		fi
	fi
fi

# swappiness: with an LLM resident in page cache we want the kernel to keep file
# cache and swap out anonymous memory earlier than the default 60. 10 is the
# usual "server with a big file cache" compromise (100 = swap aggressively,
# 0 = never swap except OOM, so 10 keeps the swapfile useful).
CURRENT_SWAPPINESS="$(cat /proc/sys/vm/swappiness)"
if [[ "$CURRENT_SWAPPINESS" == "10" ]]; then
	skip "vm.swappiness=10"
else
	sysctl -w vm.swappiness=10 >/dev/null
	echo 'vm.swappiness=10' > /etc/sysctl.d/99-tuk-convosearch.conf
	ok "vm.swappiness 10 (was $CURRENT_SWAPPINESS)"
fi

# ------------------------------------------------------------------ #
# 4. Docker daemon defaults: log caps so json logs cannot fill the disk
# ------------------------------------------------------------------ #
# deploy/docker-compose.yml already sets max-size/max-file per service; this
# daemon-wide default covers `docker run` you do by hand and any container
# added later. No rotation = a 40 GB disk gone in a week.
say "4/8 Docker daemon log driver limits"
mkdir -p /etc/docker
if [[ -f /etc/docker/daemon.json ]] && grep -q 'max-file' /etc/docker/daemon.json; then
	skip "/etc/docker/daemon.json already configures log rotation"
else
	if [[ -f /etc/docker/daemon.json ]]; then
		BACKUP="/etc/docker/daemon.json.bak.$(date +%Y%m%d%H%M%S)"
		cp /etc/docker/daemon.json "$BACKUP"
		printf "daemon.json exists but has no log rotation. It was copied to %s and will be\nreplaced with the log-rotation block below. Merge any settings you had by hand.\nOverwrite it? [Y/n] " "$BACKUP"
		read -r reply
		if [[ "${reply:-y}" =~ ^[Nn]$ ]]; then
			printf '    skipped. Containers you start by hand keep unbounded json logs.\n'
		fi
	else
		reply=y
	fi
	if [[ ! "${reply:-y}" =~ ^[Nn]$ ]]; then
		cat > /etc/docker/daemon.json <<'JSON'
{
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "10m",
    "max-file": "3"
  },
  "live-restore": true
}
JSON
		systemctl restart docker
		ok "daemon.json written; docker restarted (live-restore keeps running containers up)"
	fi
fi

# ------------------------------------------------------------------ #
# 5. Firewall: SSH + HTTP + HTTPS. Nothing else.
# ------------------------------------------------------------------ #
# The critical part is what is NOT allowed:
#   * 11434 (Ollama) — an open Ollama on a public IP is scanned within hours and
#     used to mine crypto or serve arbitrary prompts for free.
#   * 8000 (the app) — it has no IP allowlist; Caddy is the only front door, and
#     the app trusts X-Forwarded-For for rate limiting, which is only safe when
#     nothing can bypass the proxy.
say "5/8 ufw"
apt-get install -y --no-install-recommends ufw >/dev/null 2>&1 || true
ufw allow  OpenSSH      >/dev/null   # 22 — keep this first, or you lock yourself out
ufw allow  80/tcp       comment 'HTTP (also ACME challenges)'  >/dev/null
ufw allow  443/tcp      comment 'HTTPS'                        >/dev/null
# Belt and braces: if these rules ever conflict with an earlier allow, deny wins.
ufw deny  11434/tcp     comment 'Ollama must never be public'  >/dev/null
ufw deny  8000/tcp      comment 'app is reached via Caddy only' >/dev/null
ufw default deny incoming >/dev/null
ufw default allow outgoing >/dev/null
if ufw status | grep -q '^Status: active'; then
	skip "ufw already active"
else
	printf "Enable the firewall now (allow 22, 80, 443; deny everything else)? [Y/n] "
	read -r reply
	if [[ "${reply:-y}" =~ ^[Nn]$ ]]; then
		printf '%b    ufw rules are installed but NOT enabled — your VPS is wide open.%b\n' "$YELLOW" "$OFF"
	else
		ufw --force enable >/dev/null
	fi
fi
echo "     Active rules:"
ufw status numbered | sed 's/^/       /'
# Docker bypasses ufw/iptables for published ports, so a `ports: 8000:8000` in
# compose would still be reachable. Note it rather than install the famous
# docker-ufw hook: keeping the compose file free of host port mappings for
# app/ollama is the simpler, more reliable control.
printf '%b     NOTE: Docker publishes ports straight through iptables, bypassing ufw.%b\n' "$YELLOW" "$OFF"
printf '%b     Verify from another machine, not from the VPS itself:%b\n' "$YELLOW" "$OFF"
printf '       nc -zv YOUR_VPS_IP 11434   -> must fail\n'
printf '       nc -zv YOUR_VPS_IP 8000    -> must fail\n'

# ------------------------------------------------------------------ #
# 6. fail2ban — a modest sshd jail
# ------------------------------------------------------------------ #
# Unmodified Ubuntu + root password SSH receives a steady stream of bots on
# port 22. Banning after 5 failures for 1h keeps the log readable without
# banning the whole internet (and without banning you if you fat-finger it).
say "6/8 fail2ban"
apt-get install -y --no-install-recommends fail2ban >/dev/null 2>&1 || true
if [[ -f /etc/fail2ban/jail.d/tuk-convosearch.conf ]]; then
	skip "/etc/fail2ban/jail.d/tuk-convosearch.conf already present"
else
	# Using jail.local/jail.d rather than editing jail.conf: package upgrades
	# overwrite jail.conf and drop your changes.
	cat > /etc/fail2ban/jail.d/tuk-convosearch.conf <<'BAN'
[DEFAULT]
backend  = systemd
bantime  = 1h
findtime = 10m
maxretry = 5
# Your own admin machine must never be banned. Add more lines if you have
# several office/VPN IPs; a whole /24 is fine here.
ignoreip = 127.0.0.1/8 ::1

[sshd]
enabled = true
port    = ssh
mode    = normal
BAN
	systemctl enable --now fail2ban >/dev/null 2>&1 || systemctl restart fail2ban
	ok "fail2ban sshd jail installed"
fi
fail2ban-client status sshd 2>/dev/null | sed 's/^/       /' || \
	printf '%b     fail2ban-client could not query the jail; check: systemctl status fail2ban%b\n' "$YELLOW" "$OFF"

# ------------------------------------------------------------------ #
# 7. Unattended security upgrades
# ------------------------------------------------------------------ #
say "7/8 Unattended security updates"
if [[ -f /etc/apt/apt.conf.d/20auto-upgrades ]]; then
	skip "unattended-upgrades already configured"
else
	apt-get install -y --no-install-recommends unattended-upgrades >/dev/null 2>&1 || true
	# Package defaults reboot only in the maintenance window; leave that alone
	# rather than forcing reboots on a box a student is using mid-exam.
	cat > /etc/apt/apt.conf.d/20auto-upgrades <<'CONF'
APT::Periodic::Update-Package-Lists "1";
APT::Periodic::Unattended-Upgrade "1";
CONF
	ok "nightly apt metadata + security patches"
fi

# ------------------------------------------------------------------ #
# 8. Non-root deploy user
# ------------------------------------------------------------------ #
# Membership of the `docker` group is equivalent to root for anything running
# in containers. It is still worth having: daily operations, log reading and
# restarts happen as `deploy`, so a leaked session key is not the root key.
say "8/8 Deploy user '$DEPLOY_USER'"
if id "$DEPLOY_USER" >/dev/null 2>&1; then
	skip "user $DEPLOY_USER exists"
else
	adduser --disabled-password --gecos 'TUK-ConvoSearch operator' "$DEPLOY_USER"
	ok "created $DEPLOY_USER (no password set: key-only login)"
fi
usermod -aG docker "$DEPLOY_USER"
install -d -m 0755 -o "$DEPLOY_USER" -g "$DEPLOY_USER" "$APP_DIR"
ok "$APP_DIR owned by $DEPLOY_USER"

# Copy whatever key the root account already has, so day-1 login works without
# inventing a second key.
mkdir -p "/home/$DEPLOY_USER/.ssh"
chmod 700 "/home/$DEPLOY_USER/.ssh"
if compgen -G '/root/.ssh/authorized_keys' >/dev/null; then
	cp /root/.ssh/authorized_keys "/home/$DEPLOY_USER/.ssh/authorized_keys"
	chmod 644 "/home/$DEPLOY_USER/.ssh/authorized_keys"
	chown -R "$DEPLOY_USER:$DEPLOY_USER" "/home/$DEPLOY_USER/.ssh"
	ok "root's authorized_keys copied for $DEPLOY_USER"
else
	printf '%b     No /root/.ssh/authorized_keys found. Add your public key to%s\n' "$YELLOW" "$OFF"
	printf '       /home/%s/.ssh/authorized_keys before disabling passwords.\n' "$DEPLOY_USER"
fi

# ------------------------------------------------------------------ #
# Backup cron
# ------------------------------------------------------------------ #
say "Nightly backup cron"
if [[ -f /etc/cron.d/tuk-backup ]]; then
	skip "/etc/cron.d/tuk-backup already installed"
else
	printf "Install the nightly backup job (03:30 UTC as user %s)? [Y/n] " "$DEPLOY_USER"
	read -r reply
	if [[ ! "${reply:-y}" =~ ^[Nn]$ ]]; then
		# /etc/cron.d rather than a user crontab: the field format includes the
		# user, it survives `adduser`, and one file shows every job on the box.
		# deploy/tuk-backup.cron is the same content for review; the only
		# difference is that cron.d needs the username column and a PATH.
		if [[ -f "$SCRIPT_DIR/tuk-backup.cron" ]]; then
			install -m 644 "$SCRIPT_DIR/tuk-backup.cron" /etc/cron.d/tuk-backup
			ok "cron installed from $SCRIPT_DIR/tuk-backup.cron"
		else
			printf '%b     %s not found; install it later with:%b\n' "$YELLOW" "$SCRIPT_DIR/tuk-backup.cron" "$OFF"
			printf '       sudo install -m 644 deploy/tuk-backup.cron /etc/cron.d/tuk-backup\n'
		fi
	fi
fi

# ------------------------------------------------------------------ #
# Report + the manual parts
# ------------------------------------------------------------------ #
say "Summary"
cat <<SUMMARY
  Docker    : $(docker --version)
  Compose   : $(docker compose version --short 2>/dev/null || echo '?')
  Swap      : $(free -h | awk '/^Swap:/ {print $2}') used of $((SWAP_MB)) MB configured
  Firewall  : $(ufw status | head -1)
  fail2ban  : $(systemctl is-active fail2ban)
  Deploy u. : $DEPLOY_USER (in the docker group)
  App dir   : $APP_DIR

SUMMARY

say "NEXT: SSH keys, then optionally disable password login"
cat <<'KEYS'
  1. From YOUR laptop (never from the VPS), copy your public key:

       ssh-copy-id -i ~/.ssh/id_ed25519.pub deploy@YOUR_VPS_IP

     If you have no key yet:  ssh-keygen -t ed25519 -C "you@your-laptop"

  2. Prove key login works in a SECOND terminal, leaving this one open:

       ssh deploy@YOUR_VPS_IP        # must not ask for a password
       sudo docker ps               # must work without a password prompt

  3. Only once step 2 works, disable password authentication. KEEP THIS
     ORIGINAL ROOT SESSION OPEN while you do it — if you get it wrong, the
     session you are typing in is your way back in.

       # Write the drop-in (sshd reads sshd_config.d/*.conf and uses the FIRST
       # value it encounters; 60-* sorts after Ubuntu's 50-cloud-init.conf, so
       # this wins over the distro default that re-enables passwords):
       sudo tee /etc/ssh/sshd_config.d/60-hardening.conf >/dev/null <<'SSHD'
       PasswordAuthentication no
       PubkeyAuthentication yes
       KbdInteractiveAuthentication no
       PermitRootLogin prohibit-password
       MaxAuthTries 4
       SSHD

       # Check what sshd THINKS it will do — do not skip this:
       sudo sshd -T | grep -Ei 'passwordauthentication|permitrootlogin|maxauthtries'
       # EXPECTED OUTPUT:
       #   passwordauthentication no
       #   permitrootlogin prohibit-password
       #   maxauthtries 4

       # Then test a NEW login as deploy from a second terminal. Only when that
       # succeeds:
       sudo systemctl reload ssh

     Rollback if you get locked out: use the NovaHost control-panel VNC/serial
     console (it does not go through SSH), log in as root, remove
     /etc/ssh/sshd_config.d/60-hardening.conf and run `systemctl reload ssh`.

     Do not run `systemctl restart ssh` before the reload-and-test above: on
     Ubuntu the socket-activated listener means a restart can drop your
     current connection.
KEYS

say "Then continue with deploy/RUNBOOK.md, section 3"
printf 'Reboot once so the swapfile, sysctl and docker restart are exercised:\n\n'
printf '    sudo reboot\n'
printf '    ssh %s@YOUR_VPS_IP\n\n' "$DEPLOY_USER"
