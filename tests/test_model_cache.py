"""Tests for the checkpoint-metadata cache and the registry's use of it.

What is actually being tested here
----------------------------------
Two things, and neither of them is Redis.

The first is that the cache is **total**. Every method in
:mod:`app.serving.model_cache` is documented as never raising, because it sits
inside the model load path and the whole feature is an optimisation — a service
that stops answering because a cache is unreachable has invented an outage. So
the unhappy paths get most of the coverage here: no URL configured, a URL that
resolves to nothing, a record that does not parse. Each must produce a working
cache and a log line, not an exception.

The second is that the fallback dict is not a stub. It implements the same TTL
as the Redis path, because "it works without Redis" is only true if the two
behave the same, and a test that only ever exercised one of them would not
notice the day they diverged.

No test here connects to a real Redis. ``CheckpointCache(url="")`` selects
fallback mode without touching the environment, which is the seam that exists so
this suite neither needs a server nor accidentally writes into a developer's.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from app.models.config import get_model_config
from app.serving.model_cache import CheckpointCache, LoadedModel, _redact, get_checkpoint_cache
from app.serving.model_registry import ModelRegistry

CATEGORY = "bottle"
BACKEND = "patchcore"

CHECKPOINT: Path = get_model_config().checkpoint_path(BACKEND, CATEGORY)

requires_checkpoint = pytest.mark.skipif(
    not CHECKPOINT.is_file(),
    reason=f"{CHECKPOINT} not found; run `python scripts/train_patchcore.py --category {CATEGORY}`",
)


@pytest.fixture
def cache() -> CheckpointCache:
    """A cache in fallback mode, with no Redis anywhere near it."""
    return CheckpointCache(url="")


# ---------------------------------------------------------------------------
# The record
# ---------------------------------------------------------------------------


def test_record_round_trips_through_json() -> None:
    """The four fields survive the trip, including a null checkpoint path."""
    entry = LoadedModel(backend="winclip", category=CATEGORY, checkpoint_path=None, loaded_at="2026-07-28T00:00:00+00:00")

    restored = LoadedModel.from_json(entry.to_json())

    assert restored == entry
    assert restored.key == f"winclip:{CATEGORY}"


@pytest.mark.parametrize(
    "raw",
    ["not json at all", "[]", '{"category": "bottle"}', "null"],
    ids=["garbage", "wrong-type", "missing-backend", "null"],
)
def test_malformed_records_are_dropped_not_raised(raw: str) -> None:
    """Anything can write to a shared Redis; one bad key must not stop a warm-up loop."""
    assert LoadedModel.from_json(raw) is None


def test_redact_hides_a_password_and_leaves_the_rest_readable() -> None:
    """The URL is logged, and a Redis URL is a place people put passwords."""
    assert _redact("redis://joe:hunter2@cache:6379/0") == "redis://joe:***@cache:6379/0"
    assert _redact("redis://redis:6379/0") == "redis://redis:6379/0"


# ---------------------------------------------------------------------------
# The cache, without Redis
# ---------------------------------------------------------------------------


def test_records_are_readable_back(cache: CheckpointCache) -> None:
    cache.record(BACKEND, CATEGORY, str(CHECKPOINT))
    cache.record("winclip", CATEGORY, None)

    entries = {entry.key: entry for entry in cache.entries()}

    assert set(entries) == {f"{BACKEND}:{CATEGORY}", f"winclip:{CATEGORY}"}
    assert entries[f"{BACKEND}:{CATEGORY}"].checkpoint_path == str(CHECKPOINT)
    assert entries[f"winclip:{CATEGORY}"].checkpoint_path is None


def test_recording_the_same_pair_twice_leaves_one_record(cache: CheckpointCache) -> None:
    """Keyed by (backend, category), so a reload updates rather than accumulates."""
    cache.record(BACKEND, CATEGORY, "first.ckpt")
    cache.record(BACKEND, CATEGORY, "second.ckpt")

    entries = cache.entries()

    assert len(entries) == 1
    assert entries[0].checkpoint_path == "second.ckpt"


def test_forget_removes_one_record(cache: CheckpointCache) -> None:
    cache.record(BACKEND, CATEGORY, str(CHECKPOINT))
    cache.record("winclip", CATEGORY, None)

    cache.forget(BACKEND, CATEGORY)

    assert [entry.key for entry in cache.entries()] == [f"winclip:{CATEGORY}"]


def test_forgetting_something_that_was_never_recorded_is_fine(cache: CheckpointCache) -> None:
    cache.forget("efficientad", "nonexistent-category")

    assert cache.entries() == []


def test_the_fallback_expires_records_like_redis_would() -> None:
    """The TTL is not a Redis feature this leans on; it is part of the contract.

    Without this, "works without Redis" would be true only until somebody
    restarted a process that had been up for a week and re-warmed a model
    nothing had asked for since Tuesday.
    """
    cache = CheckpointCache(url="", ttl_seconds=1)
    cache.record(BACKEND, CATEGORY, str(CHECKPOINT))
    assert len(cache.entries()) == 1

    time.sleep(1.1)

    assert cache.entries() == []


def test_no_url_configured_is_fallback_mode_not_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """`make serve` on a laptop has no Redis, and that is a supported configuration."""
    monkeypatch.delenv("REDIS_URL", raising=False)
    cache = CheckpointCache(use_dotenv=False)

    cache.record(BACKEND, CATEGORY, str(CHECKPOINT))

    assert cache.mode == "memory"
    assert [entry.key for entry in cache.entries()] == [f"{BACKEND}:{CATEGORY}"]


def test_an_unreachable_redis_degrades_instead_of_raising() -> None:
    """The whole point of the module, asserted directly.

    Port 1 with nothing on it: the connection is refused immediately rather than
    hanging, so this costs a syscall rather than the socket timeout.
    """
    cache = CheckpointCache(url="redis://127.0.0.1:1/0")

    cache.record(BACKEND, CATEGORY, str(CHECKPOINT))

    assert cache.mode == "memory"
    assert [entry.key for entry in cache.entries()] == [f"{BACKEND}:{CATEGORY}"]


def test_asking_for_the_mode_does_not_connect() -> None:
    """A diagnostic that opens a socket is a diagnostic you cannot call from a health check."""
    cache = CheckpointCache(url="redis://127.0.0.1:1/0")

    assert cache.mode == "memory"


def test_get_checkpoint_cache_is_the_same_object_every_time() -> None:
    assert get_checkpoint_cache() is get_checkpoint_cache()


# ---------------------------------------------------------------------------
# What the registry does with it
# ---------------------------------------------------------------------------


def test_previously_loaded_is_empty_on_a_cold_cache(cache: CheckpointCache) -> None:
    registry = ModelRegistry(warmup=False, cache=cache)

    assert registry.previously_loaded() == []
    assert registry.warm_from_cache() == []


def test_a_record_whose_artifact_is_gone_is_dropped(tmp_path: Path, cache: CheckpointCache) -> None:
    """A deploy that replaced a checkpoint must not cost a failed load on every restart."""
    config = get_model_config().with_overrides(checkpoint_dir=tmp_path / "checkpoints")
    registry = ModelRegistry(config, exported_dir=tmp_path / "exported", warmup=False, cache=cache)
    cache.record(BACKEND, CATEGORY, str(tmp_path / "checkpoints" / "gone.ckpt"))

    warmed = registry.warm_from_cache()

    assert warmed == []
    assert cache.entries() == [], "an unloadable record should not survive the attempt"


def test_warm_from_cache_survives_a_record_it_cannot_parse(tmp_path: Path, cache: CheckpointCache) -> None:
    """One corrupt key must not cost the rest of the working set."""
    config = get_model_config().with_overrides(checkpoint_dir=tmp_path / "checkpoints")
    registry = ModelRegistry(config, exported_dir=tmp_path / "exported", warmup=False, cache=cache)
    cache.record("winclip", CATEGORY, None)

    # WinCLIP is the one backend that loads with nothing on disk, so this
    # exercises the success path without needing an artifact in the fixture.
    assert registry.warm_from_cache() == [f"winclip:{CATEGORY}"]
    assert registry.is_loaded("winclip", CATEGORY)


def test_previously_loaded_excludes_what_is_already_resident(tmp_path: Path, cache: CheckpointCache) -> None:
    """On a warm process the list is what is *missing*, so a re-warm is idempotent."""
    config = get_model_config().with_overrides(checkpoint_dir=tmp_path / "checkpoints")
    registry = ModelRegistry(config, exported_dir=tmp_path / "exported", warmup=False, cache=cache)
    registry.get_model("winclip", CATEGORY)

    assert [entry.key for entry in cache.entries()] == [f"winclip:{CATEGORY}"]
    assert registry.previously_loaded() == []


@requires_checkpoint
def test_loading_a_model_records_the_artifact_it_resolved_to(cache: CheckpointCache) -> None:
    """The recorded path is the file that was actually opened, which is what makes it useful."""
    registry = ModelRegistry(warmup=False, cache=cache)

    registry.get_model(BACKEND, CATEGORY)

    entries = cache.entries()
    assert [entry.key for entry in entries] == [f"{BACKEND}:{CATEGORY}"]
    assert entries[0].checkpoint_path == str(CHECKPOINT)


def test_a_dead_cache_does_not_stop_a_model_loading(tmp_path: Path) -> None:
    """The ordering assertion: a cache write failure cannot fail a request."""
    config = get_model_config().with_overrides(checkpoint_dir=tmp_path / "checkpoints")
    registry = ModelRegistry(
        config,
        exported_dir=tmp_path / "exported",
        warmup=False,
        cache=CheckpointCache(url="redis://127.0.0.1:1/0"),
    )

    model = registry.get_model("winclip", CATEGORY)

    assert model.model_name == "winclip"
