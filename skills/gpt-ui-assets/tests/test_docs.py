"""Offline checks that SKILL.md/README.md stay honest about the CLI: python3 tests/test_docs.py (no API calls).

C3 (docs-vs-behaviour): every `python3 $K <cmd>` recipe in SKILL.md names a real subcommand, and every backticked
--flag used in one of those recipe lines is accepted by that command's (or sub-subcommand's) argparse parser.
As a weaker, whole-document net, every other backticked --flag anywhere in SKILL.md/README.md is accepted by
*some* command in the kit (catches a typo'd or invented flag that isn't near a runnable recipe line).
"""
import re
import sys
import types
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parent
REPO_ROOT = ROOT.parent.parent  # skills/gpt-ui-assets -> skills -> repo root
SKILL_MD = ROOT / "SKILL.md"
README_MD = REPO_ROOT / "README.md"
KIT = ROOT / "scripts" / "gptkit.py"


def _build_parser():
    """Re-build main()'s argparse tree without calling parse_args (no sys.argv side effects)."""
    src = KIT.read_text()
    i = src.index("def main():")
    j = src.index("a = ap.parse_args()")
    head, main_src = src[:i], src[i:j]
    mod = types.ModuleType("gptkit_docs_probe")
    exec(compile(head, str(KIT), "exec"), mod.__dict__)
    body = main_src.split("\n", 1)[1]
    dedented = "\n".join(l[4:] if l.startswith("    ") else l for l in body.split("\n"))
    ns = dict(mod.__dict__)
    exec(compile(dedented, "<main-body>", "exec"), ns)
    return ns["sub"]  # the top-level _SubParsersAction's .choices dict of subparsers


def _flags_of(parser):
    fl = set()
    for act in parser._actions:
        fl.update(o for o in act.option_strings if o not in ("-h", "--help"))
    return fl


def _subsubparsers(parser):
    """{name: subparser} for a command with its own sub-subcommands (only `appicon` today), else {}."""
    for act in parser._actions:
        if isinstance(getattr(act, "choices", None), dict) and all(hasattr(v, "_actions") for v in act.choices.values()):
            return act.choices
    return {}


SUB = _build_parser()
ALL_FLAGS = set()
CMD_FLAGS = {}          # "forge" -> {"--out", ...}; "appicon concepts" -> {...}
for name, p in SUB.choices.items():
    fl = _flags_of(p)
    CMD_FLAGS[name] = fl
    ALL_FLAGS |= fl
    for sub_name, sp in _subsubparsers(p).items():
        sfl = _flags_of(sp)
        CMD_FLAGS[f"{name} {sub_name}"] = sfl
        ALL_FLAGS |= sfl

# $K stands for gptkit.py; recipes look like `python3 $K forge build/home ...` or
# `python3 $K appicon build icon/concept_v2.png ...`.
RECIPE_RE = re.compile(r"python3\s+\$K\s+([a-z][a-z0-9_-]*)(?:\s+([a-z][a-z0-9_-]*))?")
FLAG_RE = re.compile(r"`(--[a-z][a-zA-Z0-9-]*)")


def _recipes(text):
    """Yield (cmd_key, full_line) for every `python3 $K ...` line, cmd_key resolved against SUB/appicon."""
    for line in text.splitlines():
        m = RECIPE_RE.search(line)
        if not m:
            continue
        cmd, maybe_sub = m.group(1), m.group(2)
        if cmd == "appicon" and maybe_sub in _subsubparsers(SUB.choices["appicon"]):
            yield f"appicon {maybe_sub}", line
        else:
            yield cmd, line


def test_every_recipe_names_a_real_subcommand():
    text = SKILL_MD.read_text()
    seen = 0
    for cmd, line in _recipes(text):
        seen += 1
        base = cmd.split(" ")[0]
        assert base in SUB.choices, f"SKILL.md recipe names unknown subcommand {base!r}: {line.strip()!r}"
        if " " in cmd:
            sub_name = cmd.split(" ", 1)[1]
            assert cmd in CMD_FLAGS, f"SKILL.md recipe names unknown appicon stage {sub_name!r}: {line.strip()!r}"
    assert seen >= 20, f"only found {seen} 'python3 $K <cmd>' recipes; the extraction regex may be broken"


def test_recipe_flags_match_their_own_command():
    text = SKILL_MD.read_text()
    checked = 0
    for cmd, line in _recipes(text):
        base = cmd.split(" ")[0]
        if base not in SUB.choices:
            continue  # reported by test_every_recipe_names_a_real_subcommand
        want = CMD_FLAGS.get(cmd, CMD_FLAGS.get(base, set()))
        for flag_match in re.finditer(r"(--[a-zA-Z][a-zA-Z0-9-]*)(?:[= ]|$)", line):
            flag = flag_match.group(1)
            if flag in ("--", "--platform") and flag not in want and any(
                flag in CMD_FLAGS.get(k, set()) for k in CMD_FLAGS if k.split(" ")[0] == base
            ):
                continue
            # chrome/gpt flags quoted only as prose inside a shell comment, not gptkit's own
            if flag in ("--no-sandbox", "--window-size", "--headless", "--user-data-dir",
                        "--use-mock-keychain", "--default-background-color", "--screenshot"):
                continue
            checked += 1
            assert flag in want, (
                f"{line.strip()!r}: {flag!r} is not an argparse flag of `{cmd}` "
                f"(known flags: {sorted(want)})"
            )
    assert checked >= 30, f"only checked {checked} recipe flags; the extraction may be broken"


def test_every_backticked_flag_exists_somewhere():
    """A weaker, whole-document net: catches a flag that is a straight typo or was renamed/removed in code but
    left stale in prose (not just inside a runnable recipe line)."""
    # Chrome's own CLI flags (mentioned when documenting _chrome_flags/doctor), not gptkit's.
    chrome_flags = {"--no-sandbox", "--window-size", "--headless", "--user-data-dir",
                     "--use-mock-keychain", "--default-background-color", "--screenshot"}
    bad = []
    for path in (SKILL_MD, README_MD):
        for m in FLAG_RE.finditer(path.read_text()):
            flag = m.group(1)
            if flag in chrome_flags or flag in ALL_FLAGS:
                continue
            bad.append((path.name, flag))
    assert not bad, f"backticked flags that no command's argparse accepts: {bad}"


def test_readme_command_table_matches_skill_commands():
    """Every command named in README's phase table's "Commands" column exists in the kit (catches a
    renamed/removed command left stale in the README after SKILL.md moved on)."""
    readme = README_MD.read_text()
    table_start = readme.index("| Phase | Commands |")
    table_end = readme.index("\n\n", readme.index("| Export"))
    rows = [r for r in readme[table_start:table_end].splitlines() if r.startswith("|")][2:]  # skip header + separator
    checked = 0
    for row in rows:
        cells = [c.strip() for c in row.strip("|").split("|")]
        commands_cell = cells[1]
        for backticked in re.findall(r"`([^`]+)`", commands_cell):
            base = backticked.split()[0].split("--")[0].strip()
            if not base:
                continue  # a bare flag like `--style-ref` in the cell, not a command name
            checked += 1
            assert base in SUB.choices, f"README's command table names {backticked!r} ({base!r} is not a gptkit subcommand)"
    assert checked >= 15, f"only checked {checked} README table commands; the extraction may be broken"


if __name__ == "__main__":
    for k, fn in list(globals().items()):
        if k.startswith("test_"):
            fn()
            print("ok", k)
