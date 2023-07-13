import xarray as xr
from os.path import exists

from tools.get_list_filenames_three_months_window import get_list_filenames_three_months_window


def load_dataset_three_months_window(path_to_dir: str, month: int, train: bool = False, chunks: int = 10000) -> xr.Dataset:
    """
    Open dataset of that month from given path.
    :param path_to_dir: directory of the downloaded datasets
    :param month: Date number that represents the Julian month (1…12)
    :param train: Select whether train or test dataset is loaded
    :param chunks: load dataset in chunks of that size
    :rtype: xr.Dataset
    :return: xArray Dataset of that month
    """
    assert month > 0 and month <= 12
    ds_type = 'train' if train else 'test'
    path_to_files = get_list_filenames_three_months_window(path_to_dir, month, train)
    for p in path_to_files:
        assert exists(p)
    ds = xr.open_mfdataset(path_to_files, concat_dim='samples_dim', combine="nested", parallel=True, chunks={'samples_dim': chunks})
    return ds
