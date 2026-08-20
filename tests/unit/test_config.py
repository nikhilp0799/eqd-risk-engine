import datetime as dt

from eqdrisk.config import BaseConfig


def test_base_config_loads_from_yaml():
    cfg = BaseConfig.from_yaml("configs/base.yaml")
    assert cfg.universe.index == ["SPX"]
    assert cfg.calendar == "NYSE"
    assert cfg.canonical_snap_time == dt.time(16, 0, 0)
    assert cfg.snap_tolerance_minutes == 15
