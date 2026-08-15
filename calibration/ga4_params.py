"""Write calibration/observed_params.json from the GA4 sample.

Two paths:

  --from-bigquery   run the queries in ga4_params.sql directly (needs a GCP
                    project with billing enabled and google-cloud-bigquery
                    installed)

  --from-values     paste the numbers you read off the BigQuery console. Use
                    this if you would rather not wire up credentials; the
                    result is identical and the SQL stays the artifact.

Either path sets source="ga4", which is what unlocks calibrated mode in
config.py. Nothing else may set it.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent / "observed_params.json"
SQL = Path(__file__).resolve().parent / "ga4_params.sql"

FIELDS = [
    "baseline_session_conversion",
    "daily_users",
    "sessions_per_user_mean",
    "sessions_per_user_dispersion",
    "session_conversion_icc",
    "cuped_pre_post_correlation",
]


def _validate(vals: dict) -> None:
    p = vals["baseline_session_conversion"]
    if not 0.0 < p < 1.0:
        raise ValueError(f"baseline conversion {p} out of range")
    icc = vals["session_conversion_icc"]
    if not 0.0 <= icc < 1.0:
        raise ValueError(f"ICC {icc} out of range; check the MSW query")
    rho = vals["cuped_pre_post_correlation"]
    if not -1.0 <= rho <= 1.0:
        raise ValueError(f"correlation {rho} out of range")
    if vals["sessions_per_user_mean"] < 1.0:
        raise ValueError("mean sessions per user cannot be below 1")


def write(vals: dict, bytes_scanned: int | None = None) -> None:
    _validate(vals)
    payload = {
        "source": "ga4",
        "note": "Measured on bigquery-public-data.ga4_obfuscated_sample_ecommerce, "
                "events_20201101 to events_20210131. See ga4_params.sql.",
        "written_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "bigquery_bytes_scanned": bytes_scanned,
        **{k: vals[k] for k in FIELDS},
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {OUT}")
    for k in FIELDS:
        print(f"  {k:34s} {vals[k]}")


def from_bigquery(project: str) -> None:
    from google.cloud import bigquery  # imported lazily

    client = bigquery.Client(project=project)
    blocks = [b.strip() for b in SQL.read_text().split(";") if b.strip()]
    results, total_bytes = {}, 0

    for block in blocks:
        job = client.query(block)
        rows = list(job.result())
        total_bytes += job.total_bytes_processed or 0
        if rows:
            results.update({k: v for k, v in dict(rows[0]).items()
                            if v is not None})

    missing = [f for f in FIELDS if f not in results]
    if missing:
        raise RuntimeError(f"queries did not return: {missing}")

    vals = {f: float(results[f]) for f in FIELDS}
    vals["daily_users"] = int(round(vals["daily_users"]))
    write(vals, bytes_scanned=total_bytes)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-bigquery", metavar="PROJECT_ID")
    ap.add_argument("--from-values", action="store_true")
    for f in FIELDS:
        ap.add_argument(f"--{f.replace('_', '-')}", type=float)
    ap.add_argument("--bytes-scanned", type=int, default=None)
    args = ap.parse_args()

    if args.from_bigquery:
        from_bigquery(args.from_bigquery)
        return

    if args.from_values:
        vals = {}
        for f in FIELDS:
            v = getattr(args, f)
            if v is None:
                raise SystemExit(f"--from-values requires --{f.replace('_', '-')}")
            vals[f] = v
        vals["daily_users"] = int(round(vals["daily_users"]))
        write(vals, bytes_scanned=args.bytes_scanned)
        return

    ap.error("pass --from-bigquery PROJECT_ID or --from-values")


if __name__ == "__main__":
    main()
