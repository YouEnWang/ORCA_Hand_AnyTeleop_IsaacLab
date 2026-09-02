#!/usr/bin/env python3
"""Record ORCA UDP teleoperation packets to JSONL."""

from __future__ import annotations

import argparse
import json
import socket
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5016)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--duration", type=float, default=0.0, help="0 records until Ctrl+C.")
    args = parser.parse_args()

    args.output.expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((args.host, args.port))
    sock.settimeout(0.25)
    start = time.monotonic()
    count = 0

    print(f"Recording UDP packets from udp://{args.host}:{args.port} to {args.output}")
    try:
        with args.output.expanduser().open("w", encoding="utf-8") as file:
            while args.duration <= 0.0 or time.monotonic() - start < args.duration:
                try:
                    data, address = sock.recvfrom(65535)
                except socket.timeout:
                    continue
                record = json.loads(data.decode("utf-8"))
                record["_record_time"] = time.time()
                record["_sender"] = f"{address[0]}:{address[1]}"
                file.write(json.dumps(record, ensure_ascii=True, sort_keys=True))
                file.write("\n")
                count += 1
    except KeyboardInterrupt:
        pass
    finally:
        sock.close()
    print(f"Recorded {count} packets")


if __name__ == "__main__":
    main()

