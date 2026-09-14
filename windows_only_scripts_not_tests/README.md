# Windows-only scripts, quarantined from tests/ (2026-09-14)

These two files matched pytest's `test_*.py` collection glob but are not portable pytest
tests -- they are Windows-only scripts hardcoding `D:\Downloads\...` paths, meant to run under
`D:\conda\envs\{powerfoam,splat-distiller}\python.exe` on the original author's machine.

**`lerf_bridge_phase1_render_WINDOWS_ONLY.py` contains a real, dangerous bug on Linux**: at
MODULE TOP LEVEL (executes on import/collection, not inside any test function):

    RESULT_FOLDER = Path(rf"D:\Downloads\claude_logs\_bridge_test_{SCENE}\rendered")
    if RESULT_FOLDER.parent.exists():
        shutil.rmtree(RESULT_FOLDER.parent)

On Windows, `D:\Downloads\...` is a real absolute path and `.parent` is a legitimate scratch
folder. On Linux, backslashes are not path separators, so the whole string becomes ONE opaque
filename component; `.parent` of a single-component relative path is `Path('.')` -- the
CURRENT DIRECTORY -- which always exists. This makes the guard vacuous and the line
unconditionally executes `shutil.rmtree('.')` the instant pytest imports this file.

Confirmed by direct reproduction (2026-09-14): running `pytest tests/` (or bare `pytest`) from
a powerfoam checkout root on this Linux machine wipes the entire checkout -- deletes every file
and `.git` itself, leaving only the empty directory (the final `os.rmdir('.')` step fails with
`EINVAL`, which is the only reason the top-level directory entry survives). Reproduced twice in
one session. Do NOT move these files back into `tests/` on a Linux checkout without first
neutralising the top-level `shutil.rmtree` call.

`lerf_bridge_phase2_metrics_WINDOWS_ONLY.py` has no destructive code, just Windows-only imports
that fail to collect on Linux (harmless ImportError, not moved for safety -- moved for
consistency, since it's the same Windows-only bridge-testing pair).
