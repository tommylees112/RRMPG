from typing import Union, Any, List, Tuple, Optional, Dict
import xarray as xr 
from tqdm import tqdm
from pathlib import Path
import numpy as np 
from rrmpg.models import GR4J
import pandas as pd


def get_data_dir() -> Path:
    if Path(".").absolute().home().as_posix() == "/home/leest":
        data_dir = Path("/DataDrive200/data")
    elif Path(".").absolute().home().as_posix() == "/home/tommy":
        data_dir = Path("/datadrive/data")
    elif Path(".").absolute().home().as_posix() == "/soge-home/users/chri4118":
        data_dir = Path("/lustre/soge1/projects/crop_yield/")
    elif Path(".").absolute().home().as_posix() == "/Users/tommylees":
        data_dir = Path("/Users/tommylees/Downloads")
    else:
        assert False, "What machine are you on?"

    assert (
        data_dir.exists()
    ), f"Expect data_dir: {data_dir} to exist. Current Working Directory: {Path('.').absolute()}"
    return data_dir


def _fill_gaps_da(da: xr.DataArray, fill: Optional[str] = None, per_station: bool = True) -> xr.DataArray:
    assert isinstance(da, xr.DataArray), "Expect da to be DataArray (not dataset)"
    variable = da.name
    if fill is None:
        return da
    else:
        # fill gaps
        if fill == "median":
            # fill median
            if per_station:
                median = da.median(dim="time")
            else:
                median = da.median()
            da = da.fillna(median)
        elif fill == "interpolate":
            #  fill interpolation
            da_df = da.to_dataframe().interpolate()
            coords = [c for c in da_df.columns if c != variable]
            da = da_df.to_xarray().assign_coords(
                dict(zip(coords, da_df[coords].iloc[0].values))
            )[variable]
    return da


def fill_gaps(
    ds: Union[xr.DataArray, xr.Dataset], fill: Optional[str] = None, per_station: bool = True,
) -> Union[xr.DataArray, xr.Dataset]:
    if fill is None:
        return ds
    if isinstance(ds, xr.Dataset):
        pbar = tqdm(ds.data_vars, desc=f"Filling gaps with method {fill}")
        for v in pbar:
            pbar.set_postfix_str(v)
            ds[v] = _fill_gaps_da(ds[v], fill=fill, per_station=per_station)
    else:
        ds = _fill_gaps_da(ds, fill=fill, per_station=per_station)
    return ds

def initialise_stores(ds: xr.Dataset) -> Tuple[np.ndarray]:
    v = [v for v in ds.data_vars][0]
    qsim_data = np.empty(ds[v].shape)
    s_store_data = np.empty(ds[v].shape)
    r_store_data = np.empty(ds[v].shape)

    return (qsim_data, s_store_data, r_store_data)


def simulate_gr4j_for_one_station(ds: xr.Dataset, param_df: pd.DataFrame, station_id: int) -> xr.Dataset:
    params: Dict[str, float] = param_df.loc[station_id].to_dict()
    data = ds.sel(station_id=station_id)
    
    model = GR4J(params=params)
    qsim, s_store, r_store = model.simulate(
        prec=data["precipitation"].values, 
        etp=data["pet"].values, 
        s_init=0, 
        r_init=0, 
        return_storage=True
    )
    qsim = qsim.flatten().reshape(-1, 1)
    s_store = s_store.flatten().reshape(-1, 1)
    r_store = r_store.flatten().reshape(-1, 1)

    sim_ds = xr.Dataset(
        {
            "gr4j": (["time", "station_id"], qsim),
            "s_store": (["time", "station_id"], s_store),
            "r_store": (["time", "station_id"], r_store)
        },
        coords={"time": ds["time"], "station_id": [station_id]}
    )
    return sim_ds
