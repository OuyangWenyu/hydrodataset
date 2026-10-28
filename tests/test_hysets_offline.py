"""Offline HYSETS tests: no multi-GB OSF downloads in the default suite."""

from pathlib import Path

from hydrodataset.hysets import (
    HYSETS_DOWNLOAD_ENV,
    Hysets,
    hysets_data_dir,
    hysets_download_enabled,
    hysets_local_files_ready,
    hysets_qc_stations_path,
)


def test_hysets_download_enabled_reads_env(monkeypatch):
    """HYDRODATASET_HYSETS_DOWNLOAD opts into aqua_fetch OSF downloads."""
    monkeypatch.delenv(HYSETS_DOWNLOAD_ENV, raising=False)
    assert hysets_download_enabled() is False
    monkeypatch.setenv(HYSETS_DOWNLOAD_ENV, "1")
    assert hysets_download_enabled() is True
    monkeypatch.setenv(HYSETS_DOWNLOAD_ENV, "true")
    assert hysets_download_enabled() is True


def test_hysets_local_files_ready_requires_catalog(tmp_path, monkeypatch):
    """Missing aqua_fetch catalog files are treated as not ready."""
    monkeypatch.setattr(
        "hydrodataset.hysets.HYSETS",
        type(
            "FakeHYSETS",
            (),
            {
                "url": {
                    "HYSETS_2023_update_QC_stations.nc": "https://example.invalid/qc",
                    "HYSETS_2023_update_ERA5.nc": "https://example.invalid/era5",
                }
            },
        ),
    )
    assert hysets_local_files_ready(str(tmp_path)) is False
    data_dir = Path(hysets_data_dir(str(tmp_path)))
    data_dir.mkdir(parents=True)
    (data_dir / "HYSETS_2023_update_QC_stations.nc").write_bytes(b"qc")
    assert hysets_local_files_ready(str(tmp_path)) is False
    (data_dir / "HYSETS_2023_update_ERA5.nc").write_bytes(b"era5")
    assert hysets_local_files_ready(str(tmp_path)) is True
    assert hysets_qc_stations_path(str(tmp_path)).endswith(
        "HYSETS_2023_update_QC_stations.nc"
    )


def test_hysets_download_false_does_not_construct_aqua_fetch(tmp_path, monkeypatch):
    """Hysets(download=False) must not instantiate aqua_fetch when files are missing."""

    def _fail_hysets(uri, *args, **kwargs):
        raise AssertionError(
            f"aqua_fetch HYSETS must not be constructed for missing data: {uri}"
        )

    monkeypatch.setattr("hydrodataset.hysets.HYSETS", _fail_hysets)
    monkeypatch.setattr(
        "hydrodataset.hysets.hysets_local_files_ready", lambda uri: False
    )
    ds = Hysets(str(tmp_path), download=False)
    assert not hasattr(ds, "aqua_fetch")


def test_hysets_download_false_attaches_when_files_exist(tmp_path, monkeypatch):
    """Hysets(download=False) may construct aqua_fetch once the catalog is local."""

    class DummyHYSETS:
        def __init__(self, uri, *args, **kwargs):
            self.uri = uri

    monkeypatch.setattr("hydrodataset.hysets.HYSETS", DummyHYSETS)
    monkeypatch.setattr(
        "hydrodataset.hysets.hysets_local_files_ready", lambda uri: True
    )
    ds = Hysets(str(tmp_path), download=False)
    assert isinstance(ds.aqua_fetch, DummyHYSETS)
    assert ds.aqua_fetch.uri == str(tmp_path)


def test_hysets_download_true_constructs_aqua_fetch(tmp_path, monkeypatch):
    """Hysets(download=True) constructs aqua_fetch even if files are missing."""

    class DummyHYSETS:
        def __init__(self, uri, *args, **kwargs):
            self.uri = uri

    monkeypatch.setattr("hydrodataset.hysets.HYSETS", DummyHYSETS)
    monkeypatch.setattr(
        "hydrodataset.hysets.hysets_local_files_ready", lambda uri: False
    )
    ds = Hysets(str(tmp_path), download=True)
    assert isinstance(ds.aqua_fetch, DummyHYSETS)
