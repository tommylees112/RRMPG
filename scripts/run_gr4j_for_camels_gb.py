import xarray as xr 
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

from scripts.utils import get_data_dir, fill_gaps, initialise_stores
from rrmpg.models import CemaneigeGR4J, GR4J


if __name__ == "__main__":
    data_dir = get_data_dir()
    ds = xr.open_dataset(data_dir / "ALL_dynamic_ds.nc")

    # for each catchment in the train and test period run GR4J forward 1975--2015
    start_date = pd.Timestamp("01/01/1970")
    end_date = pd.Timestamp("31/12/2015")
    # train_start_date = pd.Timestamp("01/01/1988")
    # train_end_date = pd.Timestamp("31/12/1997")
    test_start_date = pd.Timestamp("01/01/1997")
    test_end_date = pd.Timestamp("31/12/2008")

    # ensure data is full (fill with median values - no missing values)
    ds = fill_gaps(ds, fill="median")
    test_ds = ds.sel(time=slice(test_start_date, test_end_date))

    # TODO: HOW TO SIMULATE SPATIAL VARIABILITY ? HOW TO VARY THE PARAMETERS?
    param_df = pd.read_csv(data_dir / "gr4j_params.csv").set_index("index")
    if "Unnamed: 0" in param_df.columns:
        param_df = param_df.drop(columns="Unnamed: 0")

    qsim_data, s_store_data, r_store_data = initialise_stores(ds)
    station_ids = ds.station_id.values
    for ix, station_id in enumerate(tqdm(station_ids, desc="Simulating GR4J Data")):
        params = {'x1': np.exp(5.76865628090826), 
                  'x2': np.sinh(1.61742503661094), 
                  'x3': np.exp(4.24316129943456), 
                  'x4': np.exp(-0.117506799276908)+0.5}
        
        # extract param dict from dataframe
        params = param_df.loc[station_id].to_dict()
        
        model = GR4J(params=params)

        data = ds.sel(station_id=station_id)
        qsim, s_store, r_store = model.simulate(prec=data["precipitation"].values, etp=data["pet"].values, s_init=0, r_init=0, return_storage=True)
        qsim_data[:, ix] = qsim.flatten()
        s_store_data[:, ix] = s_store.flatten()
        r_store_data[:, ix] = r_store.flatten()

    g_ds = xr.Dataset(
        {
            "gr4j": (["time", "station_id"], qsim_data),
            "s_store": (["time", "station_id"], s_store_data),
            "r_store": (["time", "station_id"], r_store_data)
        },
        coords={"time": ds.time, "station_id": station_ids}
    )
    all_ds = ds.merge(g_ds)

    # save to file
    all_ds.to_netcdf(data_dir / "GR4J_simulated_test_fitted_params.nc")

    # save station ids to file 
    fpath = Path(".").home() / "neuralhydrology/data/all_gr4j_basins_orig_grid.txt"
    np.savetxt(fpath, all_ds.station_id.values, fmt="%s")