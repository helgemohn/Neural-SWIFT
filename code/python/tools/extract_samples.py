import xarray as xr
import pandas as pd


def extract_samples(ds: xr.Dataset, num_samples: int, seed: int = 17) -> pd.DataFrame:
    """
    Extract a number of random samples from a dataset.
    :param ds: loaded dataset
    :param num_samples: number of random samples
    :param seed: random seed number
    :return: Pandas DataFrame of that number of random samples
    :rtype: pd.DataFrame
    """
    # control number of samples
    import numpy as np
    np.random.seed(seed)
    rng = np.random.default_rng()

    if num_samples > 0 and num_samples < ds.dims['samples_dim']:
        ind_samples = rng.choice(ds.dims['samples_dim'], num_samples, replace=False)
    else:
        ind_samples = rng.choice(ds.dims['samples_dim'], ds.dims['samples_dim'], replace=False)

    column_names = ds.attrs['features'].split(', ')
    return pd.DataFrame(ds['data'][ind_samples, :].to_numpy(), columns=column_names)