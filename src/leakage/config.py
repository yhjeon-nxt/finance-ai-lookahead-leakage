"""Central configuration for the look-ahead-leakage experiment.

All experiment-wide constants live here so that periods, universe, models, and paths
are defined exactly once and imported everywhere. Keeping this single-source-of-truth
also makes the eventual EC2 run reproducible from the same module.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

# --------------------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = REPO_ROOT / "results"

for _d in (RAW_DIR, PROCESSED_DIR, RESULTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# S3 (per personal-bucket policy) — used by run/ for artifact backup.
S3_BUCKET = os.environ.get("LEAKAGE_S3_BUCKET", "neuroxt-personal")
S3_PREFIX = os.environ.get("LEAKAGE_S3_PREFIX", "yhjeon/finance-ai-leakage")


# --------------------------------------------------------------------------------------
# Trading windows
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Window:
    """A trading window plus the lookback buffer needed for trailing context."""

    name: str
    start: date          # first day the agent trades
    end: date            # last day the agent trades (inclusive)
    download_start: date  # earlier than `start` to supply trailing features

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return f"{self.name} [{self.start}..{self.end}]"


# In-distribution window for the treatment model (it was trained on this era).
IN_DIST = Window(
    name="2024H2",
    start=date(2024, 7, 1),
    end=date(2024, 12, 31),
    download_start=date(2024, 3, 1),  # ~80 trading days of lookback
)

# Out-of-distribution window: postdates the treatment model's knowledge cutoff.
# Data exists as of the run date (mid-2026).
OOD = Window(
    name="2026JanMay",
    start=date(2026, 1, 1),
    end=date(2026, 5, 31),
    download_start=date(2025, 9, 1),
)

WINDOWS = {w.name: w for w in (IN_DIST, OOD)}


# --------------------------------------------------------------------------------------
# Asset universe
# --------------------------------------------------------------------------------------
# Chosen to span the marquee 2024-H2 events:
#   - 2024-08-05 yen-carry-unwind crash (broad risk-off; VIX spike)
#   - 2024-11-05 US election "Trump trade" (small caps, banks, TSLA, crypto-adjacent)
# Cash is an implicit 8th asset (target weights need not sum to 1; remainder = cash).
UNIVERSE: list[str] = [
    "SPY",   # broad market
    "QQQ",   # tech beta
    "NVDA",  # AI bellwether
    "TSLA",  # high-beta, election-sensitive
    "JPM",   # banks (Trump-trade beneficiary)
    "IWM",   # small caps (Trump-trade beneficiary)
    "COIN",  # crypto-adjacent (election + risk sentiment)
]

# Known event anchors used by the leakage/foresight metrics (date, label, direction).
# direction: -1 = sharp down-move (test pre-emptive de-risking),
#            +1 = sharp up-move  (test pre-emptive loading).
EVENT_ANCHORS = [
    (date(2024, 8, 5), "yen_carry_unwind_crash", -1),
    (date(2024, 11, 6), "us_election_trump_trade", +1),
]


# --------------------------------------------------------------------------------------
# Models / experiment groups
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class ModelSpec:
    tag: str               # ollama model tag
    documented_cutoff: str  # vendor-stated knowledge cutoff (verified empirically in Phase 1)
    label: str


TREATMENT_MODEL = ModelSpec("qwen3:8b", "2024+ (2025 release)", "treatment")
CONTROL_MODEL = ModelSpec("llama3.1:8b", "2023-12", "control")
# Tiny model purely for local pipeline smoke tests (NOT used for results).
SMOKE_MODEL = ModelSpec("llama3.2:1b", "2023-12", "smoke")


@dataclass(frozen=True)
class Group:
    name: str
    model: ModelSpec
    window: Window
    description: str


GROUPS = [
    Group("T-in", TREATMENT_MODEL, IN_DIST,
          "Treatment model on its in-distribution window (leakage candidate)."),
    Group("C-A", CONTROL_MODEL, IN_DIST,
          "Model control: pre-2024 cutoff on the same window (cannot know the future)."),
    Group("C-B", TREATMENT_MODEL, OOD,
          "Time control: treatment model on a post-cutoff window (cannot know the future)."),
]

# Experiment knobs
TRAILING_DAYS = 60        # length of price history shown to the agent at each decision
SEEDS = [0, 1, 2]         # repeats for error bars on headline comparisons
AGENT_TEMPERATURE = 0.7   # > 0 so seeds differ
STARTING_CASH = 1_000_000.0
