"""Map Kepler-plan rows to items by star, tier and period (item ids can change when items are
rebuilt; the labelled signal is a star, a tier and a period).

A plan row maps to the item of the same Gaia ID and tier whose period is nearest, if within 1%.
"""


def item_index(items):
    """items: dict of columns with gaia_dr3, tier, period. Returns a lookup function."""
    by = {}
    for i, (g, t, p) in enumerate(zip(items["gaia_dr3"], items["tier"], items["period"])):
        by.setdefault((int(g), "box" if t == "box" else "sine"), []).append((float(p), i))

    def find(gaia, tier, period):
        best = None
        for p, i in by.get((int(gaia), tier), []):
            d = abs(p - period) / period
            if d < 0.01 and (best is None or d < best[0]):
                best = (d, i)
        return None if best is None else best[1]
    return find


def plan_to_items(plan, items):
    """Index into items for each plan row (None where no item matches)."""
    find = item_index(items)
    return [find(g, t, p) if k == "signal" else None
            for g, t, p, k in zip(plan["gaia_dr3"], plan["tier"], plan["period"], plan["kind"])]


