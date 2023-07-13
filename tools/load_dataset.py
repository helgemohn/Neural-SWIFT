import xarray as xr
from os.path import exists


def load_dataset(path_to_dir: str, month: int, train: bool = False, chunks: int = 10000) -> xr.Dataset:
    """
    Open dataset of that month from given path.
    :rtype: xr.Dataset
    :param path_to_dir: directory of the downloaded datasets
    :param month: Date number that represents the Julian month (1…12)
    :param train: Select whether train or test dataset is loaded
    :param chunks: load dataset in chunks of that size
    :return: xArray Dataset of that month
    """
    assert month > 0 and month <= 12
    ds_type = 'train' if train else 'test'
    path_to_file = path_to_dir + '/SWIFT-AI_{}_month_{}.nc'.format(ds_type, str(month).zfill(2))
    assert exists(path_to_file)
    ds = xr.open_dataset(path_to_file, chunks=chunks)

    return ds
