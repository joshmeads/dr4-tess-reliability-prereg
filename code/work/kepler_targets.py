"""Kepler targets with long-cadence light curves: KIC IDs from the public stpubdata bucket's
directory listing (kepler/public/lightcurves/XXXX/XXXXXXXXX/). Listing only; no files fetched.
Output: data/kepler_targets.txt (one KIC per line)."""
import pathlib
import re
import time
import urllib.parse
import urllib.request

D = pathlib.Path(__file__).resolve().parents[1] / "data"
BASE = "https://stpubdata.s3.amazonaws.com/?list-type=2&delimiter=/&prefix="
UA = {"User-Agent": "dr4-reliability/1.0 (python-urllib)"}


def prefixes(prefix):
    out, token = [], None
    while True:
        url = BASE + urllib.parse.quote(prefix) + (f"&continuation-token={urllib.parse.quote(token)}" if token else "")
        for k in range(4):
            try:
                body = urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120).read().decode()
                break
            except OSError:
                if k == 3:
                    raise
                time.sleep(10)
        out += re.findall(r"<Prefix>([^<]+/)</Prefix>", body)
        m = re.search(r"<NextContinuationToken>([^<]+)</NextContinuationToken>", body)
        if not m:
            return [p for p in out if p != prefix]
        token = m.group(1)


def main():
    tops = prefixes("kepler/public/lightcurves/")
    kics = []
    for i, p in enumerate(tops):
        kics += [int(q.rstrip("/").split("/")[-1]) for q in prefixes(p)]
        if i % 20 == 0:
            print(f"{i}/{len(tops)} groups, {len(kics)} targets", flush=True)
    (D / "kepler_targets.txt").write_text("\n".join(map(str, sorted(set(kics)))) + "\n")
    print("Kepler targets with light curves:", len(set(kics)))


if __name__ == "__main__":
    main()
