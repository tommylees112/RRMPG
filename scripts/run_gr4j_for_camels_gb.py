import xarray as xr 
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

from scripts.utils import get_data_dir, fill_gaps
from rrmpg.models import CemaneigeGR4J, GR4J


if __name__ == "__main__":
    data_dir = get_data_dir()
    ds = xr.open_dataset(data_dir / "ALL_dynamic_ds.nc")

    # for each catchment in the train and test period run GR4J forward 1975--2015
    start_date = pd.Timestamp("01/01/1970")
    end_date = pd.Timestamp("31/12/2015")

    # ensure data is full (fill with median values - no missing values)
    ds = fill_gaps(ds, fill="median")

    # TODO: HOW TO SIMULATE SPATIAL VARIABILITY ? HOW TO VARY THE PARAMETERS?
    v = [v for v in ds.data_vars][0]
    data = np.empty(ds[v].shape)
    station_ids = ds.station_id.values
    for ix, station_id in enumerate(tqdm(station_ids, desc="Simulating GR4J Data")):
        params = {'x1': np.exp(5.76865628090826), 
                  'x2': np.sinh(1.61742503661094), 
                  'x3': np.exp(4.24316129943456), 
                  'x4': np.exp(-0.117506799276908)+0.5}
        model = GR4J(params=params)

        qsim = model.simulate(prec=ds["precipitation"], etp=ds["pet"], s_init=0, r_init=0)
        data[:, ix] = qsim
        break
