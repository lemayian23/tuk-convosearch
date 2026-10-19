#!/usr/bin/env python
"""
Create the first admin account.

    python scripts/create_admin.py --email you@tuk.ac.ke --name "Denis Kirionki"

There is no default password and no seeded account on purpose: an assistant
that indexes university documents should not boot with a credential that is
printed in a startup log and identical on every deployment.

Location: scripts/create_admin.py
"""

import argparse
import getpass
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services import database  # noqa: E402
from app.services.auth import hash_password, password_policy_error  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Create an admin account")
    parser.add_argument("--email", required=True)
    parser.add_argument("--name", default="Administrator")
    parser.add_argument("--password", help="Omit to be prompted (recommended)")
    args = parser.parse_args()

    database.init_db()

    existing = database.get_user_by_email(args.email)
    if existing:
        print(f"An account already exists for {args.email}.")
        return 1

    password = args.password or getpass.getpass("Admin password: ")
    error = password_policy_error(password)
    if error:
        print(f"Refused: {error}")
        return 1

    user_id = database.create_user(
        full_name=args.name,
        email=args.email,
        password_hash=hash_password(password),
        role="admin",
    )
    print(f"Created admin #{user_id} for {args.email}. Store the password somewhere safe.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
