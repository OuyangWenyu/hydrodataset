import pytest

from hydrodataset import StandardVariable
from hydrodataset.camelsh import Camelsh
from hydrodataset.camels_lux import CamelsLux
from hydrodataset.lamah_ce import LamahCe
from hydrodataset.lamah_ice import LamahIce


@pytest.mark.parametrize(
    ("dataset_cls", "expected_unit"),
    [
        pytest.param(CamelsLux, "mm/day", id="camels-lux-daily"),
        pytest.param(Camelsh, "mm/h", id="camelsh-hourly"),
        pytest.param(LamahCe, "mm/day", id="lamah-ce-daily"),
        pytest.param(LamahIce, "mm/day", id="lamah-ice-daily"),
    ],
)
def test_precipitation_sources_use_cache_timestep_unit(dataset_cls, expected_unit):
    sources = dataset_cls._dynamic_variable_mapping[
        StandardVariable.PRECIPITATION
    ]["sources"]

    assert sources
    assert {source["unit"] for source in sources.values()} == {expected_unit}
