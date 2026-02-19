# analysis.py
# Build exactly the same outputs as 01_build_master_2025.ipynb:
# 1) master_2025_raw_valid.csv.gz
# 2) master_2025_joblevel_submission.csv.gz
# 3) master_2025_joblevel_terminal.csv.gz

from pathlib import Path
import re
import gzip
import shutil
import random

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


# =============================
# 0) Paths (fit your folder)
# =============================
PROJECT_ROOT = Path(__file__).resolve().parent

# Input: prefer .txt.gz, otherwise .txt
INPUT_GZ = PROJECT_ROOT / "whole_cluster_usage_merged_raw.txt.gz"
INPUT_TXT = PROJECT_ROOT / "whole_cluster_usage_merged_raw.txt"

if INPUT_GZ.exists():
    INPUT_PATH = INPUT_GZ
    INPUT_COMPRESSION = "gzip"
elif INPUT_TXT.exists():
    INPUT_PATH = INPUT_TXT
    INPUT_COMPRESSION = None
else:
    raise FileNotFoundError(
        f"Cannot find input file. Expected one of:\n- {INPUT_GZ}\n- {INPUT_TXT}"
    )

OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CHUNK_SIZE = 200_000
BUCKET_COUNT = 256
RANDOM_SEED = 427

RAW_VALID_PATH = OUTPUT_DIR / "master_2025_raw_valid.csv.gz"
SUBMISSION_PATH = OUTPUT_DIR / "master_2025_joblevel_submission.csv.gz"
TERMINAL_PATH = OUTPUT_DIR / "master_2025_joblevel_terminal.csv.gz"
TMP_BUCKET_DIR = OUTPUT_DIR / "_tmp_joblevel_buckets"

print("PROJECT_ROOT:", PROJECT_ROOT)
print("INPUT_PATH:", INPUT_PATH)
print("OUTPUT_DIR:", OUTPUT_DIR)
print("CHUNK_SIZE:", f"{CHUNK_SIZE:,}")
print("BUCKET_COUNT:", BUCKET_COUNT)

# Reset outputs (match notebook behavior)
for p in [RAW_VALID_PATH, SUBMISSION_PATH, TERMINAL_PATH]:
    if p.exists():
        p.unlink()
if TMP_BUCKET_DIR.exists():
    shutil.rmtree(TMP_BUCKET_DIR)


# =============================
# 1) Same cleaning helpers as notebook
# =============================
NA_TOKENS = {"", "unknown", "none", "n/a"}
REQMEM_PATTERN = re.compile(r"^\s*([0-9]*\.?[0-9]+)\s*([kKmMgGtT])(?:[cCnN])?\s*$")

RAW_OUTPUT_COLUMNS = [
    "JobIDRaw", "JobID", "JobName",
    "User", "Group", "Account",
    "Partition", "QOS",
    "Submit", "Eligible", "Start", "End",
    "State", "Reason", "ExitCode",
    "ReqCPUS", "ReqMem", "ReqMem_MB", "ReqNodes",
    "AllocCPUS", "AllocNodes", "AllocTRES",
    "ElapsedRaw", "TimelimitRaw",
    "WaitTimeSec", "RunTimeSec",
]

normalize_cols = [
    "JobIDRaw", "JobID", "JobName", "User", "Group", "Account",
    "Partition", "QOS", "Submit", "Eligible", "Start", "End", "State",
    "Reason", "ExitCode", "ReqCPUS", "ReqMem", "ReqNodes",
    "AllocCPUS", "AllocNodes", "AllocTRES", "ElapsedRaw", "TimelimitRaw",
]

def normalize_missing(series: pd.Series) -> pd.Series:
    s = series.astype("string").str.strip()
    return s.mask(s.str.lower().isin(NA_TOKENS))

def parse_reqmem_mb(series: pd.Series) -> pd.Series:
    s = normalize_missing(series)
    extracted = s.str.extract(REQMEM_PATTERN)
    numeric = pd.to_numeric(extracted[0], errors="coerce")
    unit = extracted[1].str.upper()
    factor = unit.map({"K": 1/1024, "M": 1, "G": 1024, "T": 1024*1024})
    return numeric * factor

def count_input_data_lines(path: Path) -> int:
    # Match notebook: count data lines (no header) for bad-line estimation
    if path.suffix == ".gz":
        with gzip.open(path, mode="rt", encoding="utf-8", errors="replace") as f:
            _ = next(f, None)
            return sum(1 for _ in f)
    else:
        with open(path, mode="rt", encoding="utf-8", errors="replace") as f:
            _ = next(f, None)
            return sum(1 for _ in f)


# =============================
# 2) Build master_2025_raw_valid.csv.gz (exactly)
# =============================
print("\n[Step 1] Validate header column count...")
header_df = pd.read_csv(
    INPUT_PATH,
    sep="|",
    compression=INPUT_COMPRESSION,
    engine="python",
    nrows=0,
)
header_cols = header_df.columns.tolist()
print("Header column count:", len(header_cols))
if len(header_cols) != 120:
    raise RuntimeError(f"Expected 120 columns, got {len(header_cols)}. Check input formatting.")

print("[Step 1] Counting input lines (for bad-line estimation)...")
input_data_lines = count_input_data_lines(INPUT_PATH)
print(f"Input data lines (without header): {input_data_lines:,}")

parsed_rows = 0
written_rows = 0
header_written = False

reader = pd.read_csv(
    INPUT_PATH,
    sep="|",
    compression=INPUT_COMPRESSION,
    engine="python",
    chunksize=CHUNK_SIZE,
    on_bad_lines="skip",
    dtype="string",
)

for chunk_idx, chunk in enumerate(tqdm(reader, desc="Building raw_valid"), start=1):
    parsed_rows += len(chunk)

    # normalize selected columns (exactly like notebook)
    for c in normalize_cols:
        if c in chunk.columns:
            chunk[c] = normalize_missing(chunk[c])

    # keep only 2025 by Submit year (exactly)
    chunk = chunk[chunk["Submit"].str[:4].eq("2025").fillna(False)].copy()
    if chunk.empty:
        continue

    # State first token (exactly)
    chunk["State"] = normalize_missing(chunk["State"]).str.split().str[0]

    # ReqMem_MB (exactly)
    chunk["ReqMem_MB"] = parse_reqmem_mb(chunk["ReqMem"])

    # WaitTimeSec & RunTimeSec (exactly)
    submit_ts = pd.to_datetime(chunk["Submit"], errors="coerce")
    start_ts = pd.to_datetime(chunk["Start"], errors="coerce")
    end_ts = pd.to_datetime(chunk["End"], errors="coerce")
    chunk["WaitTimeSec"] = (start_ts - submit_ts).dt.total_seconds()
    chunk["RunTimeSec"] = (end_ts - start_ts).dt.total_seconds()

    out_chunk = chunk.reindex(columns=RAW_OUTPUT_COLUMNS)

    out_chunk.to_csv(
        RAW_VALID_PATH,
        mode="a",
        index=False,
        header=not header_written,
        compression="gzip",
    )
    header_written = True
    written_rows += len(out_chunk)

    if chunk_idx % 20 == 0:
        print(f"  Chunk {chunk_idx}: parsed_rows={parsed_rows:,}, written_rows={written_rows:,}")

bad_line_estimate = max(input_data_lines - parsed_rows, 0)
filter_drop_estimate = parsed_rows - written_rows

print("\n[Step 1] Summary")
print(f"parsed_rows (after parser, bad lines skipped): {parsed_rows:,}")
print(f"written_rows (2025 raw_valid): {written_rows:,}")
print(f"estimated_bad_lines_skipped: {bad_line_estimate:,}")
print(f"estimated_rows_dropped_by_2025_filter: {filter_drop_estimate:,}")
print(f"raw_valid output: {RAW_VALID_PATH}")


# =============================
# 3) Build job-level submission & terminal (exactly)
# =============================
print("\n[Step 2] Bucketize rows for job-level dedup...")

TMP_BUCKET_DIR.mkdir(parents=True, exist_ok=True)
for p in TMP_BUCKET_DIR.glob("bucket_*.csv.gz"):
    p.unlink()

joblevel_work_cols = [
    "JobIDRaw", "JobID", "JobName",
    "User", "Group", "Account",
    "Partition", "QOS",
    "Submit", "Eligible", "Start", "End",
    "State", "Reason", "ExitCode",
    "ReqCPUS", "ReqMem_MB", "ReqNodes",
    "AllocCPUS", "AllocNodes", "AllocTRES",
    "ElapsedRaw", "TimelimitRaw",
    "WaitTimeSec", "RunTimeSec",
]

submission_cols = [
    "JobIDRaw",
    "User", "Group", "Account",
    "Partition", "QOS",
    "Submit",
    "State",
    "ReqCPUS", "ReqMem_MB", "ReqNodes",
    "AllocCPUS", "AllocNodes",
]

terminal_cols = [
    "JobIDRaw",
    "User", "Group", "Account",
    "Partition", "QOS",
    "Submit",
    "State",
    "ReqCPUS", "ReqMem_MB", "ReqNodes",
    "AllocCPUS", "AllocNodes",
    "Start", "End", "ElapsedRaw", "ExitCode", "Reason",
    "WaitTimeSec", "RunTimeSec",
]

row_cursor = 0
bucket_rows = 0

reader = pd.read_csv(
    RAW_VALID_PATH,
    compression="gzip",
    chunksize=CHUNK_SIZE,
)

for chunk in tqdm(reader, desc="Bucketizing for job-level"):
    chunk = chunk.reindex(columns=joblevel_work_cols).copy()

    # keep valid JobIDRaw (exactly)
    chunk = chunk[chunk["JobIDRaw"].notna()].copy()
    chunk["JobIDRaw"] = chunk["JobIDRaw"].astype("string").str.strip()
    chunk = chunk[chunk["JobIDRaw"].ne("")].copy()

    if chunk.empty:
        continue

    n = len(chunk)
    chunk["_row_order"] = np.arange(row_cursor, row_cursor + n, dtype=np.int64)
    row_cursor += n

    bucket_ids = (
        pd.util.hash_pandas_object(chunk["JobIDRaw"], index=False).astype("uint64") % BUCKET_COUNT
    ).astype("int64")
    chunk["_bucket"] = bucket_ids.values

    for bucket_id, sub in chunk.groupby("_bucket", sort=False):
        path = TMP_BUCKET_DIR / f"bucket_{int(bucket_id):03d}.csv.gz"
        sub.drop(columns=["_bucket"]).to_csv(
            path,
            mode="a",
            index=False,
            header=not path.exists(),
            compression="gzip",
        )
        bucket_rows += len(sub)

print(f"Bucketized rows: {bucket_rows:,}")
print(f"Bucket directory: {TMP_BUCKET_DIR}")


# ---- submission: earliest submit within each JobIDRaw (exactly) ----
print("\n[Step 3] Build submission job-level table (exactly)...")

header_written = False
submission_jobs = 0
bucket_files = sorted(TMP_BUCKET_DIR.glob("bucket_*.csv.gz"))

if SUBMISSION_PATH.exists():
    SUBMISSION_PATH.unlink()

for path in tqdm(bucket_files, desc="Dedup submission per bucket"):
    bdf = pd.read_csv(path, compression="gzip")
    if bdf.empty:
        continue

    bdf["_row_order"] = pd.to_numeric(bdf["_row_order"], errors="coerce").fillna(10**18)
    bdf["_SubmitTS"] = pd.to_datetime(bdf["Submit"], errors="coerce")
    bdf["_SubmitTS_sort"] = bdf["_SubmitTS"].fillna(pd.Timestamp.max)

    bdf = bdf.sort_values(
        ["JobIDRaw", "_SubmitTS_sort", "_row_order"],
        ascending=[True, True, True],
        kind="mergesort",
    )

    selected = bdf.drop_duplicates(subset=["JobIDRaw"], keep="first")
    out = selected.reindex(columns=submission_cols)

    out.to_csv(
        SUBMISSION_PATH,
        mode="a",
        index=False,
        header=not header_written,
        compression="gzip",
    )
    header_written = True
    submission_jobs += len(out)

print(f"submission unique jobs: {submission_jobs:,}")
print(f"submission output: {SUBMISSION_PATH}")


# ---- terminal: priority End > Start > Submit, latest timekey, exact tie-breakers ----
print("\n[Step 4] Build terminal job-level table (exactly)...")

header_written = False
terminal_jobs = 0

if TERMINAL_PATH.exists():
    TERMINAL_PATH.unlink()

for path in tqdm(bucket_files, desc="Dedup terminal per bucket"):
    bdf = pd.read_csv(path, compression="gzip")
    if bdf.empty:
        continue

    bdf["_row_order"] = pd.to_numeric(bdf["_row_order"], errors="coerce").fillna(10**18)

    bdf["_SubmitTS"] = pd.to_datetime(bdf["Submit"], errors="coerce")
    bdf["_StartTS"] = pd.to_datetime(bdf["Start"], errors="coerce")
    bdf["_EndTS"] = pd.to_datetime(bdf["End"], errors="coerce")

    # priority: 2 (End exists), 1 (Start exists only), 0 (Submit only)
    bdf["_priority"] = np.select(
        [bdf["_EndTS"].notna(), bdf["_StartTS"].notna()],
        [2, 1],
        default=0,
    )

    bdf["_TimeKey"] = bdf["_EndTS"].where(
        bdf["_EndTS"].notna(),
        bdf["_StartTS"].where(bdf["_StartTS"].notna(), bdf["_SubmitTS"]),
    )
    bdf["_TimeKey_sort"] = bdf["_TimeKey"].fillna(pd.Timestamp("1900-01-01"))

    bdf = bdf.sort_values(
        ["JobIDRaw", "_priority", "_TimeKey_sort", "_row_order"],
        ascending=[True, False, False, True],
        kind="mergesort",
    )

    selected = bdf.drop_duplicates(subset=["JobIDRaw"], keep="first")
    out = selected.reindex(columns=terminal_cols)

    out.to_csv(
        TERMINAL_PATH,
        mode="a",
        index=False,
        header=not header_written,
        compression="gzip",
    )
    header_written = True
    terminal_jobs += len(out)

print(f"terminal unique jobs: {terminal_jobs:,}")
print(f"terminal output: {TERMINAL_PATH}")

print("\nALL DONE.")