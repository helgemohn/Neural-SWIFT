import xarray as xr
import pandas as pd

def big_data_description(data: xr.Dataset, variables: list) -> pd.DataFrame:
    """
    Calculate statistics of the Xarray Dataset.
    :param ds: loaded dataset
    :param num_samples: number of random samples
    :param seed: random seed number
    :return: Pandas DataFrame of that number of random samples
    :rtype: pd.DataFrame
    
    Copyright Helge Mohn 2023
    """
    variables_all = data.attrs['features'].replace('[', '').replace(']', '').replace("'", '').split(', ')

    # get indices of selected variables
    ind_variables = []
    for var in variables:
        ind_variables.append(variables_all.index(var))
        #print("{} - {} - {}".format(var, ind_variables[-1], variables_all[ind_variables[-1]]))

    data = data['data'][:, ind_variables]    
    
    import numpy as np
    coords = np.arange(0, len(variables))

    xr_mean = xr.DataArray(data.mean(dim='samples_dim'), 
                 coords=[coords], 
                 dims=["features"],
                 name="mean",
                 attrs={"features":str(variables),}) 

    xr_std = xr.DataArray(data.std(dim='samples_dim'), 
                 coords=[coords], 
                 dims=["features"],
                 name="stddev",
                 attrs={"features":str(variables),}) 

    xr_min = xr.DataArray(data.min(dim='samples_dim'), 
                 coords=[coords], 
                 dims=["features"],
                 name="min",
                 attrs={"features":str(variables),}) 

    xr_max = xr.DataArray(data.max(dim='samples_dim'), 
                 coords=[coords], 
                 dims=["features"],
                 name="max",
                 attrs={"features":str(variables),}) 

    xr_median = xr.DataArray(data.median(dim='samples_dim'), 
                 coords=[coords], 
                 dims=["features"],
                 name="median",
                 attrs={"features":str(variables),}) 

    tmp = xr.merge( [xr_mean, xr_std, xr_min, xr_max, xr_median], compat='override', combine_attrs='override', )
    
    tmp = tmp.to_pandas()
    tmp['variables'] = variables
    tmp = tmp.set_index(tmp['variables']).drop('variables', axis=1)
    return tmp.transpose()
