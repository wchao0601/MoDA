#!/usr/bin/env python3
"""Audit whether every ISPRS checkpoint can be deserialized by this checkout.

Each checkpoint is loaded in a fresh process.  This is intentional: failed pickle
imports and large model objects must not contaminate subsequent checks.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


PROBE = r"""
import json
import sys
import traceback
from pathlib import Path

result = {"weight": sys.argv[1], "ok": False}
try:
    from ultralytics import YOLO
    model = YOLO(sys.argv[1])
    net = model.model
    state = net.state_dict()
    result.update(
        ok=True,
        task=getattr(model, "task", None),
        model_class=f"{type(net).__module__}.{type(net).__qualname__}",
        tensors=len(state),
        parameters=sum(v.numel() for v in state.values()),
        yaml=getattr(net, "yaml", None),
    )
except Exception as exc:
    result.update(
        error_type=type(exc).__name__,
        error=str(exc),
        traceback=traceback.format_exc(),
    )
print("__ISPRS_AUDIT__" + json.dumps(result, ensure_ascii=False, default=str))
"""


def checkpoint_paths(root: Path, include_last: bool) -> list[Path]:
    names = {"best.pt", "last.pt"} if include_last else {"best.pt"}
    return sorted(p for p in root.rglob("*.pt") if p.name in names)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("ISPRS-weights"))
    parser.add_argument("--output", type=Path, default=Path("isprs_weight_audit.json"))
    parser.add_argument("--include-last", action="store_true")
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()

    root = args.root.resolve()
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", "/tmp/msrodet-matplotlib")
    env.setdefault("YOLO_CONFIG_DIR", "/tmp/msrodet-ultralytics")
    results = []
    for index, weight in enumerate(checkpoint_paths(root, args.include_last), 1):
        relative = weight.relative_to(root).as_posix()
        print(f"[{index}] {relative}", flush=True)
        try:
            proc = subprocess.run(
                [sys.executable, "-c", PROBE, str(weight)],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=args.timeout,
                env=env,
            )
            marker = next(
                (line.removeprefix("__ISPRS_AUDIT__") for line in proc.stdout.splitlines() if line.startswith("__ISPRS_AUDIT__")),
                None,
            )
            item = json.loads(marker) if marker else {
                "weight": str(weight),
                "ok": False,
                "error_type": "ProbeProtocolError",
                "error": f"probe exited {proc.returncode} without a result marker",
            }
            item["weight"] = relative
            item["returncode"] = proc.returncode
            item["stderr_tail"] = proc.stderr[-4000:]
        except subprocess.TimeoutExpired as exc:
            item = {"weight": relative, "ok": False, "error_type": "TimeoutExpired", "error": str(exc)}
        results.append(item)
        print("  OK" if item["ok"] else f"  FAIL: {item['error_type']}: {item['error']}", flush=True)

    payload = {
        "root": str(root),
        "checkpoint_count": len(results),
        "passed": sum(item["ok"] for item in results),
        "failed": sum(not item["ok"] for item in results),
        "results": results,
    }
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: payload[key] for key in ("checkpoint_count", "passed", "failed")}, indent=2))
    return bool(payload["failed"])


if __name__ == "__main__":
    raise SystemExit(main())
