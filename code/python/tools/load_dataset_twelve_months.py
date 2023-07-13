import xarray as xr
from os.path import exists

from tools.get_list_filenames_twelve_months import get_list_filenames_twelve_months


def load_dataset_twelve_months(path_to_dir: str, train: bool = False, chunks: int = 10000) -> xr.Dataset:
    """
    Open dataset of that month from given path.
    :param path_to_dir: directory of the downloaded datasets
    :param train: Select whether train or test dataset is loaded
    :param chunks: load dataset in chunks of that size
    :rtype: xr.Dataset
    :return: xArray Dataset of that month
    """
    assert exists(path_to_dir)

    ds_type = 'train' if train else 'test'
    path_to_files = get_list_filenames_twelve_months(path_to_dir, train)

    ds = xr.open_mfdataset(path_to_files, concat_dim='samples_dim', combine="nested", parallel=True, chunks={'samples_dim': chunks})
    return ds
