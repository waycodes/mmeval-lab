#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import yaml  # type: ignore
except Exception as e:
    print("Missing PyYAML. Install with: python -m pip install pyyaml", file=sys.stderr)
    raise

ID_RE = re.compile(r"\b[a-zA-Z0-9_-]+-[a-zA-Z0-9]+(?:\.[0-9]+)*\b")

@dataclass(frozen=True)
class IssueSpec:
    key: str
    parent: Optional[str]
    type: str
    priority: int
    title: str
    description: str

@dataclass(frozen=True)
class DepSpec:
    type: str
    from_key: str
    to_key: str

def run(cmd: List[str]) -> str:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed ({p.returncode}): {' '.join(cmd)}\n{p.stdout}")
    return p.stdout.strip()

def ensure_bd() -> None:
    try:
        out = run(["bd", "--version"])
        print(f"[ok] bd present: {out}")
    except Exception as e:
        raise RuntimeError("bd not found. Install Beads and ensure `bd` is on PATH.") from e

def ensure_init() -> None:
    # Heuristic: check .beads directory; if absent, init.
    if not Path(".beads").exists():
        print("[info] .beads missing; running `bd init`")
        run(["bd", "init"])

def parse_created_id(output: str) -> str:
    # bd create often prints the id in output; extract the first plausible id token.
    m = ID_RE.search(output)
    if not m:
        raise RuntimeError(f"Could not parse issue id from output:\n{output}")
    return m.group(0)

def bd_create(title: str, issue_type: str, priority: int, description: str) -> str:
    # Keep to widely used flags: -t, -p, --description (seen in public usage guides).
    cmd = [
        "bd", "create", title,
        "-t", issue_type,
        "-p", str(priority),
        "--description", description,
    ]
    out = run(cmd)
    issue_id = parse_created_id(out)
    return issue_id

def bd_dep_add(dep_type: str, from_id: str, to_id: str) -> None:
    # Dep direction: FROM blocks TO. (So for "A depends on B", you add dep from B -> A.)
    cmd = ["bd", "dep", "add", from_id, to_id]
    # Many installs support --type; if yours does not, delete these two args.
    cmd += ["--type", dep_type]
    run(cmd)

def load_specs(path: Path) -> Tuple[IssueSpec, List[IssueSpec], List[DepSpec]]:
    data = yaml.safe_load(path.read_text())
    root = data["project"]["root"]
    root_spec = IssueSpec(
        key=root["key"],
        parent=None,
        type=root["type"],
        priority=int(root["priority"]),
        title=root["title"],
        description=root.get("description", ""),
    )
    issues = []
    for it in data.get("issues", []):
        issues.append(IssueSpec(
            key=it["key"],
            parent=it.get("parent"),
            type=it.get("type", "task"),
            priority=int(it.get("priority", 2)),
            title=it["title"],
            description=it.get("description", ""),
        ))
    deps = []
    for d in data.get("dependencies", []):
        deps.append(DepSpec(
            type=d["type"],
            from_key=d["from"],
            to_key=d["to"],
        ))
    return root_spec, issues, deps

def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: scripts/import_beads.py .beads/mmevallab.yaml", file=sys.stderr)
        sys.exit(2)

    ensure_bd()
    ensure_init()

    root_spec, issues, deps = load_specs(Path(sys.argv[1]))

    created: Dict[str, str] = {}

    # 1) Create ROOT epic.
    print("[create] ROOT epic")
    created[root_spec.key] = bd_create(
        title=root_spec.title,
        issue_type=root_spec.type,
        priority=root_spec.priority,
        description=root_spec.description,
    )
    print(f"  {root_spec.key} -> {created[root_spec.key]}")

    # 2) Create all issues (no dependencies yet).
    for spec in issues:
        print(f"[create] {spec.key}")
        created[spec.key] = bd_create(
            title=spec.title,
            issue_type=spec.type,
            priority=spec.priority,
            description=spec.description,
        )
        print(f"  {spec.key} -> {created[spec.key]}")

    # 3) Wire parent-child hierarchy.
    # bd dep add expects: child first, then parent for parent-child type
    for spec in issues:
        if spec.parent:
            parent_id = created[spec.parent]
            child_id = created[spec.key]
            print(f"[dep parent-child] {spec.parent} -> {spec.key}")
            bd_dep_add("parent-child", child_id, parent_id)

    # 4) Wire explicit ordering deps.
    # bd dep add expects: dependency first, then dependent (B blocks A means add B A)
    for d in deps:
        from_id = created[d.from_key]
        to_id = created[d.to_key]
        print(f"[dep {d.type}] {d.from_key} -> {d.to_key}")
        bd_dep_add(d.type, to_id, from_id)

    print("[done] Import complete.")
    print("Next: bd dep tree <ROOT_ID>  and  bd ready")

if __name__ == "__main__":
    main()
