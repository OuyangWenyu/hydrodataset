"""Shared pytest fixtures and markers for the hydrodataset test suite."""

import os
import pytest


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "skip_if_ci: skip test that requires large dataset download (not suitable for CI)",
    )
    config.addinivalue_line(
        "markers",
        "integration: test that requires real dataset data on disk",
    )


# ── Shared marker ──────────────────────────────────────────────────────

skip_if_ci = pytest.mark.skipif(
    bool(os.getenv("CI")),
    reason="Requires large dataset download, not suitable for CI",
)
