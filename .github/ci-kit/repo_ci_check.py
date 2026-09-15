#!/usr/bin/env python3
"""repo_ci_check.py - CI/CD best-practice checker for a GitHub repository (CI-PROTOCOL).

Usage: repo_ci_check.py [REPO_DIR] [--json] [--quiet]
Exit codes: 0 all required checks pass . 1 one or more FAIL . 2 usage/not-a-repo

Stdlib only; regex-based (no PyYAML) so it runs on any host and inside CI.
Checks (industry sources listed in CI-PROTOCOL.md):
  WF   .github/workflows/*.yml exists
  TRIG a workflow triggers on pull_request AND push
  PERM every workflow declares permissions: (least-privilege GITHUB_TOKEN)
  PIN  every `uses:` carries an immutable ref (tag or SHA; never main/master/latest)
  SHA  (warn) refs are full 40-hex SHAs
  DEP  .github/dependabot.yml covers package-ecosystem github-actions
  CONC (warn) concurrency: with cancel-in-progress
  TMO  (warn) every job sets timeout-minutes
  STEP at least one workflow step runs a command (tests/lint) - CI does work
  DOCS README.md and .gitignore exist; LICENSE (warn)
  DRIFT repos with generated manifests (manifest.config.json / FUNCTION_MANIFEST.md /
        GENERATOR_MANIFEST.tsv) must run a drift gate in CI (reproducibility enforced)
"""
import glob
import json
import os
import re
import sys

MUTABLE_REFS = {"main", "master", "latest", "develop", "dev", "trunk", "HEAD"}
SHA_RE = re.compile(r"^[0-9a-f]{40}$")
USES_RE = re.compile(r"^\s*-?\s*uses:\s*['\"]?([^'\"\s#]+)", re.M)
MANIFEST_MARKERS = ("manifest.config.json", "FUNCTION_MANIFEST.md", "GENERATOR_MANIFEST.tsv")


def _read(p):
    with open(p, encoding="utf-8", errors="replace") as fh:
        return fh.read()


def _strip_comments(text):
    return "\n".join(line.split(" #")[0] if not line.lstrip().startswith("#") else "" for line in text.splitlines())


def _top_keys(text):
    return {m.group(1) for m in re.finditer(r"^([A-Za-z_][\w-]*):", text, re.M)}


def _job_blocks(text):
    """Return list of (job_name, block_text) for jobs: children (2-space indented keys)."""
    m = re.search(r"^jobs:\s*$", text, re.M)
    if not m:
        return []
    body = text[m.end():]
    parts = re.split(r"^  ([A-Za-z_][\w-]*):\s*$", body, flags=re.M)
    out = []
    for i in range(1, len(parts) - 1, 2):
        out.append((parts[i], parts[i + 1]))
    return out


def check(repo):
    results = []  # (level, code, message)

    def fail(code, msg):
        results.append(("FAIL", code, msg))

    def warn(code, msg):
        results.append(("WARN", code, msg))

    def ok(code, msg):
        results.append(("PASS", code, msg))

    wf_dir = os.path.join(repo, ".github", "workflows")
    wfs = sorted(glob.glob(os.path.join(wf_dir, "*.yml")) + glob.glob(os.path.join(wf_dir, "*.yaml")))
    if not wfs:
        fail("WF", "no workflow under .github/workflows/ (need at least one CI workflow)")
    else:
        ok("WF", "%d workflow file(s)" % len(wfs))

    has_pr = has_push = has_step = drift_wired = False
    for wf in wfs:
        rel = os.path.relpath(wf, repo).replace(os.sep, "/")
        raw = _read(wf)
        text = _strip_comments(raw)
        keys = _top_keys(text)
        on_block = re.search(r"^(?:on|'on'|\"on\"):(.*?)(?=^[A-Za-z_][\w-]*:|\Z)", text, re.M | re.S)
        on_txt = on_block.group(0) if on_block else ""
        if re.search(r"pull_request", on_txt):
            has_pr = True
        if re.search(r"\bpush\b", on_txt):
            has_push = True
        if "permissions" in keys:
            ok("PERM", "%s: top-level permissions" % rel)
        else:
            jobs = _job_blocks(text)
            if jobs and all(re.search(r"^\s{4}permissions:", b, re.M) for _, b in jobs):
                ok("PERM", "%s: per-job permissions" % rel)
            else:
                fail("PERM", "%s: no permissions: block (set least-privilege GITHUB_TOKEN, e.g. contents: read)" % rel)
        if "concurrency" in keys:
            ok("CONC", "%s: concurrency set" % rel)
        else:
            warn("CONC", "%s: add concurrency: {group: ${{ github.workflow }}-${{ github.ref }}, cancel-in-progress: true}" % rel)
        jobs = _job_blocks(text)
        missing_tmo = [n for n, b in jobs if not re.search(r"^\s{4}timeout-minutes:", b, re.M)]
        if jobs and not missing_tmo:
            ok("TMO", "%s: all jobs have timeout-minutes" % rel)
        elif missing_tmo:
            warn("TMO", "%s: jobs without timeout-minutes: %s" % (rel, ", ".join(missing_tmo)))
        if re.search(r"^\s*-?\s*run:", text, re.M):
            has_step = True
        for m in USES_RE.finditer(text):
            ref = m.group(1)
            if ref.startswith("./") or ref.startswith("docker://"):
                continue
            if "@" not in ref:
                fail("PIN", "%s: uses %s has no @ref (pin to a release tag or full SHA)" % (rel, ref))
                continue
            action, tag = ref.rsplit("@", 1)
            if tag in MUTABLE_REFS:
                fail("PIN", "%s: uses %s pinned to mutable ref %s" % (rel, action, tag))
            elif not re.match(r"^[A-Za-z0-9][A-Za-z0-9._/-]*$", tag) or "__" in tag:
                fail("PIN", "%s: uses %s@%s is not a valid ref (unrendered placeholder?)" % (rel, action, tag))
            elif not SHA_RE.match(tag):
                warn("SHA", "%s: uses %s@%s is a tag, not a full commit SHA (immutable only when SHA-pinned)" % (rel, action, tag))
        if re.search(r"drift[-_]gate|check-generated-manifests-clean", text):
            drift_wired = True
    if wfs:
        if has_pr and has_push:
            ok("TRIG", "workflows trigger on pull_request and push")
        else:
            fail("TRIG", "need a workflow that triggers on BOTH pull_request and push (missing: %s)"
                 % ", ".join(x for x, v in (("pull_request", has_pr), ("push", has_push)) if not v))
        if has_step:
            ok("STEP", "at least one run: step (CI does work)")
        else:
            fail("STEP", "no run: step in any workflow - CI must lint/test/build something")

    dep = os.path.join(repo, ".github", "dependabot.yml")
    if os.path.isfile(dep) and re.search(r"package-ecosystem:\s*['\"]?github-actions", _read(dep)):
        ok("DEP", ".github/dependabot.yml covers github-actions")
    else:
        fail("DEP", "missing .github/dependabot.yml with package-ecosystem: github-actions")

    for name, level in (("README.md", "FAIL"), (".gitignore", "FAIL"), ("LICENSE", "WARN")):
        present = any(os.path.isfile(os.path.join(repo, n)) for n in (name, name.lower(), name + ".md", name + ".txt"))
        if present:
            ok("DOCS", "%s present" % name)
        elif level == "FAIL":
            fail("DOCS", "%s missing" % name)
        else:
            warn("DOCS", "%s missing (add one for any repo that may be shared)" % name)

    generated = []
    for root, dirs, files in os.walk(repo):
        dirs[:] = [d for d in dirs if d not in (".git", "node_modules", ".venv", "venv", "vendor", "upstream")]
        for f in files:
            if f in MANIFEST_MARKERS:
                generated.append(os.path.relpath(os.path.join(root, f), repo).replace(os.sep, "/"))
        if len(generated) > 3:
            break
    if generated:
        if drift_wired:
            ok("DRIFT", "generated manifests present and a drift gate runs in CI")
        else:
            fail("DRIFT", "generated manifests present (%s) but no workflow runs drift-gate.sh - reproducibility is unenforced"
                 % ", ".join(generated[:3]))
    else:
        ok("DRIFT", "no generated manifests detected (no drift gate required)")
    return results


def main(argv):
    args = [a for a in argv if not a.startswith("--")]
    flags = {a for a in argv if a.startswith("--")}
    repo = os.path.abspath(args[0] if args else ".")
    if not os.path.isdir(repo):
        print("[repo_ci_check] not a directory: %s" % repo, file=sys.stderr)
        return 2
    results = check(repo)
    fails = [r for r in results if r[0] == "FAIL"]
    if "--json" in flags:
        print(json.dumps({"repo": repo, "ok": not fails,
                          "results": [{"level": l, "code": c, "message": m} for l, c, m in results]}, indent=1))
    elif "--quiet" not in flags:
        for l, c, m in results:
            print("%s %-5s %s" % (l, c, m))
        print("[repo_ci_check] %s: %d FAIL, %d WARN (repo=%s)" % ("FAIL" if fails else "PASS", len(fails),
              sum(1 for r in results if r[0] == "WARN"), repo))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
