"""
config_store.py
----------------
Simple env/`.env`-only settings. No UI, no config.json - these are the
knobs an operator sets once via environment, not something end users touch:

    DOCINTEL_BASE_URL               default: http://localhost:8000
    DOCINTEL_TIMEOUT                default: 180 (seconds, per HTTP call)
    DOCINTEL_MAX_PARALLEL_UPLOADS   default: 3   (concurrent upload+extract pipelines)

Auth is NOT here - it comes from an interactive login (see auth.py /
app.py), not from an env var. There is no long-lived token to configure.
"""
import os

from dotenv import load_dotenv

# Loads variables from a .env file (if present) in the current working
# directory or any parent directory into os.environ. Without this,
# os.environ.get() only sees variables actually exported into the shell -
# a .env file on disk does nothing on its own.
load_dotenv()


def get_base_url() -> str:
    return os.environ.get("DOCINTEL_BASE_URL", "http://localhost:8000")


def get_timeout() -> int:
    try:
        return int(os.environ.get("DOCINTEL_TIMEOUT", "180"))
    except ValueError:
        return 180


def get_max_parallel_uploads() -> int:
    try:
        n = int(os.environ.get("DOCINTEL_MAX_PARALLEL_UPLOADS", "3"))
        return max(1, n)
    except ValueError:
        return 3