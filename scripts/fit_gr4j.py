import xarray as xr 
import pandas as pd
import numpy as np
from tqdm import tqdm

from scripts.utils import get_data_dir, fill_gaps
from rrmpg.models import CemaneigeGR4J, GR4J


if __name__ == "__main__":
    data_dir = get_data_dir()
    ds = xr.open_dataset(data_dir / "ALL_dynamic_ds.nc")

    # for each catchment in the train and test period run GR4J forward 1975--2015
    train_start_date = pd.Timestamp("01/01/1988")
    train_end_date = pd.Timestamp("31/12/1997")

    # ensure data is full (fill with median values - no missing values)
    ds = fill_gaps(ds, fill="median")
    train_ds = ds.sel(time=slice(train_start_date, train_end_date))

    # for each station fit the model
    pbar = tqdm(train_ds.station_id.values, desc="Fitting GR4J")
    all_params = np.empty((train_ds.station_id.size, 4))
    
    for ix, station_id in enumerate(pbar):
        pbar.set_postfix_str(station_id)
        data = train_ds.sel(station_id=station_id)
        init_params = {'x1': np.exp(5.76865628090826), 
                    'x2': np.sinh(1.61742503661094), 
                    'x3': np.exp(4.24316129943456), 
                    'x4': np.exp(-0.117506799276908)+0.5}
        model = GR4J(params=init_params)

        res = model.fit(data["discharge_spec"].values, data["precipitation"].values, data["pet"].values)
        params = res.x
        all_params[ix] = params

    param_names = [f"x{i+1}" for i in range(4)]
    param_df = pd.DataFrame(all_params, columns=param_names, index=train_ds.station_id.values)

    # save the parameters
    param_df.reset_index().rename({"Unnamed: 0": "station_id"}, axis=1).to_csv(data_dir / "gr4j_params.csv")

