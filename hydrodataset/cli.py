"""Command-line interface for hydrodataset.

A thin Typer layer over the public API (``resolve_data_path`` / ``open_dataset``
and the reader methods).  It contains no data logic of its own — every command
just parses arguments, calls the library, and formats the output.

Examples
--------
    hydrodataset list
    hydrodataset info camels_us
    hydrodataset ids camels_us --limit 10
    hydrodataset resolve camels_us --source cloud
    hydrodataset read-ts camels_us --gages 01013500 --vars precipitation,streamflow
    hydrodataset read-attr camels_us --gages 01013500 --vars area,p_mean
    hydrodataset config
"""

from __future__ import annotations

from typing import Optional

import typer

app = typer.Typer(
    add_completion=True,
    no_args_is_help=True,
    help="hydrodataset command-line interface (local NC / cloud zarr).",
)

# ---- shared option types --------------------------------------------------
SourceOpt = typer.Option(
    None,
    "--source",
    "-s",
    help="Storage backend: 'local' or 'cloud'. Omit to use storage.default_source.",
)


def _split(csv: Optional[str]) -> Optional[list]:
    """Turn a comma-separated option into a list (or None)."""
    if not csv:
        return None
    return [x.strip() for x in csv.split(",") if x.strip()]


def _fail(msg: str) -> None:
    typer.secho(f"Error: {msg}", fg=typer.colors.RED, err=True)
    raise typer.Exit(code=1)


def _write_or_show(ds, out: Optional[str]) -> None:
    """Write an xarray Dataset to NC/CSV, or print a summary to stdout."""
    if out is None:
        typer.echo(ds)
        return
    low = out.lower()
    if low.endswith(".nc"):
        ds.to_netcdf(out)
    elif low.endswith(".csv"):
        ds.to_dataframe().to_csv(out)
    else:
        _fail(f"unsupported output extension for '{out}' (use .nc or .csv)")
    typer.secho(f"written: {out}", fg=typer.colors.GREEN)


# ---- commands -------------------------------------------------------------
@app.command("list")
def list_datasets() -> None:
    """List all registered datasets (id, reader, module.class)."""
    from hydrodataset.configs.data_resolver import READER_ALIASES, _DEFAULT_REGISTRY

    typer.echo(f"{'dataset_id':<16} {'reader':<16} module.class")
    typer.echo("-" * 70)
    for ds_id in sorted(_DEFAULT_REGISTRY):
        reader = _DEFAULT_REGISTRY[ds_id].get("reader", "")
        alias = READER_ALIASES.get(reader, {})
        cls = f"{alias.get('module','?')}.{alias.get('class','?')}"
        typer.echo(f"{ds_id:<16} {reader:<16} {cls}")
    typer.echo(f"\n{len(_DEFAULT_REGISTRY)} datasets")


@app.command()
def resolve(
    dataset: str = typer.Argument(..., help="Dataset id, e.g. camels_us."),
    source: Optional[str] = SourceOpt,
) -> None:
    """Print the resolved absolute path / S3 URI for a dataset."""
    from hydrodataset import resolve_data_path
    from hydrodataset.configs.data_resolver import DatasetResolutionError

    try:
        typer.echo(resolve_data_path(dataset, source=source))
    except DatasetResolutionError as e:
        _fail(str(e))


@app.command()
def info(
    dataset: str = typer.Argument(..., help="Dataset id."),
    source: Optional[str] = SourceOpt,
) -> None:
    """Show resolved path, default time range, station count and features."""
    from hydrodataset import resolve_data_path, open_dataset
    from hydrodataset.configs.data_resolver import DatasetResolutionError

    try:
        uri = resolve_data_path(dataset, source=source)
        ds = open_dataset(dataset, source=source)
    except DatasetResolutionError as e:
        _fail(str(e))

    typer.echo(f"dataset : {dataset}")
    typer.echo(f"source  : {source or '(default)'}")
    typer.echo(f"uri     : {uri}")
    try:
        typer.echo(f"t_range : {ds.default_t_range}")
    except Exception:
        pass
    try:
        ids = ds.read_object_ids()
        typer.echo(f"stations: {len(ids)}")
    except Exception as e:
        typer.echo(f"stations: <unavailable: {e}>")
    try:
        typer.echo(f"static features  : {list(ds.available_static_features)}")
    except Exception:
        pass
    try:
        typer.echo(f"dynamic features : {list(ds.available_dynamic_features)}")
    except Exception:
        pass


@app.command()
def ids(
    dataset: str = typer.Argument(..., help="Dataset id."),
    source: Optional[str] = SourceOpt,
    limit: int = typer.Option(0, "--limit", "-n", help="Max IDs to print (0 = all)."),
) -> None:
    """List station/gauge IDs for a dataset."""
    from hydrodataset import open_dataset
    from hydrodataset.configs.data_resolver import DatasetResolutionError

    try:
        ds = open_dataset(dataset, source=source)
        station_ids = [str(x) for x in ds.read_object_ids()]
    except DatasetResolutionError as e:
        _fail(str(e))
    shown = station_ids if limit <= 0 else station_ids[:limit]
    for s in shown:
        typer.echo(s)
    typer.echo(f"\n{len(station_ids)} stations (showing {len(shown)})", err=True)


@app.command("read-ts")
def read_ts(
    dataset: str = typer.Argument(..., help="Dataset id."),
    source: Optional[str] = SourceOpt,
    gages: Optional[str] = typer.Option(None, "--gages", "-g", help="Comma-separated station ids (default: all)."),
    variables: Optional[str] = typer.Option(None, "--vars", "-v", help="Comma-separated standard variable names (default: all dynamic)."),
    t_range: Optional[str] = typer.Option(None, "--t-range", "-t", help="Time range 'START,END' (default: dataset default)."),
    out: Optional[str] = typer.Option(None, "--out", "-o", help="Output file (.nc or .csv); omit to print summary."),
) -> None:
    """Read timeseries data (standardized variable names)."""
    from hydrodataset import open_dataset
    from hydrodataset.configs.data_resolver import DatasetResolutionError

    try:
        ds = open_dataset(dataset, source=source)
        ts = ds.read_ts_xrdataset(
            gage_id_lst=_split(gages),
            t_range=_split(t_range),
            var_lst=_split(variables),
        )
    except DatasetResolutionError as e:
        _fail(str(e))
    except Exception as e:
        _fail(str(e))
    _write_or_show(ts, out)


@app.command("read-attr")
def read_attr(
    dataset: str = typer.Argument(..., help="Dataset id."),
    source: Optional[str] = SourceOpt,
    gages: Optional[str] = typer.Option(None, "--gages", "-g", help="Comma-separated station ids (default: all)."),
    variables: Optional[str] = typer.Option(None, "--vars", "-v", help="Comma-separated standard attribute names (default: all static)."),
    out: Optional[str] = typer.Option(None, "--out", "-o", help="Output file (.nc or .csv); omit to print summary."),
) -> None:
    """Read static attribute data (standardized names)."""
    from hydrodataset import open_dataset
    from hydrodataset.configs.data_resolver import DatasetResolutionError

    try:
        ds = open_dataset(dataset, source=source)
        var_lst = _split(variables) or list(ds.available_static_features)
        attr = ds.read_attr_xrdataset(gage_id_lst=_split(gages), var_lst=var_lst)
    except DatasetResolutionError as e:
        _fail(str(e))
    except Exception as e:
        _fail(str(e))
    _write_or_show(attr, out)


@app.command()
def config() -> None:
    """Show the effective storage configuration (secrets masked)."""
    from hydrodataset.configs.settings import (
        get_storage_config,
        get_default_source,
        get_local_root,
        get_cache_dir,
        get_s3_config,
        DEFAULT_SETTING_PATH,
    )

    typer.echo(f"settings file : {DEFAULT_SETTING_PATH}")
    try:
        typer.echo(f"default_source: {get_default_source()}")
        typer.echo(f"local.root    : {get_local_root()}")
        typer.echo(f"cache dir     : {get_cache_dir()}")
        s3 = get_s3_config() or {}
        typer.echo("s3:")
        for k in ("bucket", "prefix", "endpoint_url"):
            typer.echo(f"  {k:<16}: {s3.get(k)}")
        for k in ("access_key_id", "secret_access_key"):
            v = s3.get(k)
            typer.echo(f"  {k:<16}: {'***set***' if v else '(unset)'}")
    except Exception as e:
        _fail(str(e))


if __name__ == "__main__":
    app()
