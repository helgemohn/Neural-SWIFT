import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch
from typing import Optional

###############################
#     LightningDataModule     #
###############################
features = ["modeldate", "dayofyear", "month", "year", "overhead", "eqlat", "sza", "daylight", "O2_reaction_coef", "O3_reaction_coef", "NOx_reaction_coef", "NOy_reaction_coef", "ClOy_reaction_coef", "ClOx_reaction_coef", "longitude", "latitude", "z", "temperature", "theta", "pv", "Cly", "Bry", "NOy", "HOy", "Ox", "dOx", "dCly", "dBry", "dNOy", "dHOy"]


class SwiftAI_DataModule(pl.LightningDataModule):
    def __init__(self, hparams, features_in: list, features_out=['dOx'], *args, **kwargs):
        super().__init__()

        self.batch_size = hparams.batch_size
        self.num_train = int(hparams.num_traindata)
        self.num_val = int(hparams.num_validata)
        self.num_test = int(hparams.num_testdata)
        self.num_workers = int(hparams.num_workers)
        self.features_in = features_in
        self.features_out = features_out
        self.dim_input = hparams.dim_input
        self.dim_output = hparams.dim_output

        self.data_path_train = hparams.data_path_train
        self.data_path_test = hparams.data_path_test
        self.data_meanstddev = hparams.data_path_meanstddev

        # load mean and stddev
        import xarray as xr
        xr_dataset = xr.open_dataset(self.data_meanstddev)
        import ast
        features_meanstddev = ast.literal_eval(xr_dataset['mean'].attrs['features'])
        features_sel = self.features_in.copy()
        features_sel.extend(self.features_out)
        import pandas as pd
        mean = pd.DataFrame(xr_dataset['mean'].values.reshape(1,-1), columns=features_meanstddev)
        std = pd.DataFrame(xr_dataset['stddev'].values.reshape(1,-1), columns=features_meanstddev)
        self.mean = mean[features_sel].values[0].tolist()
        self.std = std[features_sel].values[0].tolist()

    def setup(self, stage: Optional[str] = None):
        import xarray as xr

        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            # Note: in our case we load the pickle-file and create a tensor dataset
            # that is stored within the instance of the Lightning module
            # Train / Validation

            # open xarray dataset
            HDF5_USE_FILE_LOCKING = False
            data = xr.load_dataset(self.data_path_train, chunks={'samples_dim':1000000})

            # get indices of input and output features
            ind_features = []
            for feat in self.features_in:
                ind_features.append(features.index(feat))
            for feat in self.features_out:
                ind_features.append(features.index(feat))

            # control number of samples
            import numpy as np
            rng = np.random.default_rng()

            if self.num_train > 0 and self.num_train < data.sizes['samples_dim']:
                ind_samples = rng.choice(data.sizes['samples_dim'], self.num_train, replace=False)
            else:
                ind_samples = rng.choice(data.sizes['samples_dim'], data.sizes['samples_dim'], replace=False)

            data = data['data'][ind_samples, ind_features]

            # normalization
            data[:, :] = (data[:, :] - self.mean) / self.std

            X = data.sel(features_dim=slice(0, self.dim_input))
            y = data.sel(features_dim=slice(self.dim_input, self.dim_input+self.dim_output))

            # create TensorDataset
            tensor_x = torch.Tensor(X.values)
            tensor_y = torch.Tensor(y.values)
            tensor_ds = TensorDataset(tensor_x, tensor_y)   

            # Split and subset
            self.num_train = len(tensor_ds) - self.num_val
            data_train, data_val = random_split(tensor_ds, [self.num_train, self.num_val])
            # assign to use in dataloaders
            self.data_train = data_train
            self.data_val = data_val
            print("Data loaded (train {:.2e}, val {:.2e})".format(self.num_train, self.num_val))

        # Assign test dataset for use in dataloader(s)
        elif stage == 'test' or stage == 'predict':
            # open xarray dataset
            HDF5_USE_FILE_LOCKING = False
            data = xr.load_dataset(self.data_path_test, chunks={'samples_dim':1000000})

            # get indices of input and output features
            ind_features = []
            for feat in self.features_in:
                ind_features.append(features.index(feat))
            for feat in self.features_out:
                ind_features.append(features.index(feat))

            # control number of samples
            import numpy as np
            rng = np.random.default_rng()

            if self.num_test > 0 and self.num_test < data.sizes['samples_dim']:
                ind_samples = rng.choice(data.sizes['samples_dim'], self.num_test, replace=False)
            else:
                ind_samples = rng.choice(data.sizes['samples_dim'], data.sizes['samples_dim'], replace=False)

            data = data['data'][ind_samples, ind_features]
            print(data.shape)

            # normalization
            data[:, :] = (data[:, :] - self.mean) / self.std

            X = data.sel(features_dim=slice(0, self.dim_input))
            y = data.sel(features_dim=slice(self.dim_input, self.dim_input+self.dim_output))

            # create TensorDataset
            tensor_x = torch.Tensor(X.values)
            tensor_y = torch.Tensor(y.values)
            tensor_ds = TensorDataset(tensor_x, tensor_y)

            # assign to use in dataloaders
            self.data_test = tensor_ds
            print("Data loaded (test {:.2e})".format(len(data)))

    def train_dataloader(self):
        """train set removes a subset to use for validation"""
        loader = DataLoader(
            self.data_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
        )
        return loader

    def val_dataloader(self):
        """val set uses a subset of the training set for validation"""
        loader = DataLoader(
            self.data_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
        )
        return loader        

    def test_dataloader(self):
        """test set uses the test split"""
        loader = DataLoader(
            self.data_test,
            batch_size=1000000,#self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
        )
        print(loader)
        return loader

    def predict_dataloader(self):
        """test set uses the test split"""
        loader = DataLoader(
            self.data_test,
            batch_size=1000000,#len(self.data_test),
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False,
            pin_memory=True,
        )
        return loader