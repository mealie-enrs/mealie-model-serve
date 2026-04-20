#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import swiftclient
from keystoneauth1 import session as ks_session
from keystoneauth1.identity import v3 as v3_identity


def _swift_conn(args: argparse.Namespace) -> swiftclient.Connection:
    if args.auth_url and args.app_credential_id and args.app_credential_secret:
        auth = v3_identity.ApplicationCredential(
            auth_url=args.auth_url,
            application_credential_id=args.app_credential_id,
            application_credential_secret=args.app_credential_secret,
        )
        sess = ks_session.Session(auth=auth)
        return swiftclient.Connection(session=sess)

    raise SystemExit(
        "Swift credentials missing. Provide --auth-url, --app-credential-id, and --app-credential-secret."
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--container", required=True)
    parser.add_argument("--key", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--auth-url", required=True)
    parser.add_argument("--app-credential-id", required=True)
    parser.add_argument("--app-credential-secret", required=True)
    args = parser.parse_args()

    conn = _swift_conn(args)
    _, data = conn.get_object(args.container, args.key)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(data)
    print(output)


if __name__ == "__main__":
    main()
