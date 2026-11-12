"""Locations outside this directory, from environment variables (no local paths in the code).

DR4REL_SURVEY_STATE  the survey's private state directory: its SQLite ledger (survey.db), the
                     stored light curves (store/) and the cached VSX table (cache/vsx_table.parquet).
Only the scripts that build the frozen inputs need it; the published data replace it downstream.
"""
import os
import pathlib


def survey_state():
    v = os.environ.get("DR4REL_SURVEY_STATE")
    if not v:
        raise SystemExit("set DR4REL_SURVEY_STATE to the survey's state directory")
    return pathlib.Path(v)
