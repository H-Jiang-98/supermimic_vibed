"""Source the configured activation script, then run a repository script."""

import argparse
import os
from pathlib import Path
import shutil
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from supermimic.config import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--config", help="JSON config path (default: SUPERMIMIC_CONFIG or config.json)")
    parser.add_argument("script", help="Python or shell script; relative to the repository")
    parser.add_argument("args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if any(arg == "--config" or arg.startswith("--config=") for arg in args.args):
        parser.error("Place --config before the script/run name so it also selects the Conda environment")
    settings = load_config(args.config)
    activation = Path(settings["conda_env"])
    if not activation.is_file() or not os.access(activation, os.R_OK):
        parser.error(f"conda_env must point to a readable activation script: {activation}")
    bash = shutil.which("bash")
    if not bash:
        parser.error("Bash is not available on PATH")
    script = (REPO / args.script).resolve()
    if not script.is_file() or script.suffix not in (".py", ".sh"):
        parser.error(f"Expected an existing .py or .sh script: {script}")
    env = os.environ.copy()
    env["SUPERMIMIC_CONFIG"] = settings["config_file"]
    env.setdefault("WARP_CACHE_PATH", str(REPO / ".cache/warp"))
    env.setdefault("TORCHINDUCTOR_CACHE_DIR", str(REPO / ".cache/inductor"))
    env.setdefault("MPLCONFIGDIR", str(REPO / ".cache/matplotlib"))
    os.chdir(REPO)
    interpreter = "python" if script.suffix == ".py" else "bash"
    command = [interpreter]
    if interpreter == "python":
        command.append("-u")
    # Pass paths/arguments as positional parameters, never interpolate them into shell code.
    # A function keeps task arguments out of the sourced script's positional parameters.
    activate_and_run = """sm_activation_script=$1
sm_repo_dir=$2
shift 2
sm_activate_environment() { source "$sm_activation_script"; }
sm_activate_environment
sm_activation_status=$?
if [ "$sm_activation_status" -ne 0 ]; then exit "$sm_activation_status"; fi
cd "$sm_repo_dir"
exec "$@"
"""
    os.execvpe(bash, [bash, "--noprofile", "--norc", "-e", "-c", activate_and_run,
                     "supermimic", str(activation), str(REPO), *command, str(script), *args.args], env)


if __name__ == "__main__":
    main()
