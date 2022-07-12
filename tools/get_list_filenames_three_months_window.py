from pathlib import Path
from os.path import exists


def get_list_filenames_three_months_window(path_to_dir: str, month: int, train: bool = False) -> []:
    """
    Get a list of filenames for a time-window of a certain number of months.
    :param path_to_dir: directory of the downloaded datasets
    :param month: Date number that represents the Julian month (1…12)
    :param train: Select whether train or test dataset is loaded
    :return: list of filenames for this three month
    """
    assert exists(path_to_dir)
    assert month > 0 and month <= 12

    monthstr = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    p = Path(path_to_dir)

    month_before = (month - 1 - 1) % 12 + 1
    month_middle = month
    month_after = (month - 1 + 1) % 12 + 1

    filenames = []
    ds_type = 'train' if train else 'test'
    path_to_file = path_to_dir + "SWIFT-AI-DS_{}".format(ds_type) + "_month_{}.nc"

    for month_idx in {month_before, month_middle, month_after}:
        month_digit = str(month_idx).zfill(2)
        fn = path_to_file.format(month_digit)
        assert exists(fn)
        filenames.append(fn)

    return filenames
