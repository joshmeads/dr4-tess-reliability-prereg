"""Freeze the survey's identity-stage verdict counts (the basis of the E2 prior) as a public JSON.

The identity stage checks each eb item against catalogued variable stars nearby and closes it as
OFF_TARGET when a brighter catalogued variable within the TESS aperture explains the signal.
Counts only (distinct items per verdict); no identifiers. Reads the ledger read-only.
Output: results/identity_counts.json. Needs DR4REL_SURVEY_STATE.
"""
import json
import pathlib
import sqlite3
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from paths import survey_state  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]


def main():
    con = sqlite3.connect(f"file:{survey_state() / 'survey.db'}?mode=ro", uri=True)
    counts = dict(con.execute("SELECT verdict, COUNT(DISTINCT item_id) FROM verdict WHERE stage='identity' GROUP BY 1"))
    con.close()
    (ROOT / "results" / "identity_counts.json").write_text(json.dumps(dict(
        description="Distinct eb items per identity-stage verdict, snapshot 2026-11-11.", counts=counts), indent=1))
    print(counts)


if __name__ == "__main__":
    main()
