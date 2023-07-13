import pytorch_lightning as pl
from argparse import ArgumentParser
import torch
import torch.nn as nn
from torch.nn import functional as F
import math

# Copyright https://github.com/jayroxis/qres/blob/master/QRes/discussion/sin_wave/models.py
# QRes Layer: inspired by https://pytorch.org/docs/stable/_modules/torch/nn/modules/linear.html#Linear
class QResLayer(nn.Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: torch.Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super(QResLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_1 = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_2 = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight_1, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.weight_2, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_1)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        h_1 = F.linear(input, self.weight_1, bias=None)
        h_2 = F.linear(input, self.weight_2, bias=None)
        return torch.add(
            torch.mul(h_1, h_2),
            F.linear(input, self.weight_1, self.bias)
        )

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


torch.nn.QResLayer = QResLayer


# Helge Mohn (2021)
class DeepNeuralNet(pl.LightningModule):
    def __init__(self, hparams, *args, **kwargs):
        super().__init__()

        # call this to save params:
        self.hparams = hparams
        self.save_hyperparameters()

        self.learning_rate = hparams.learning_rate
        self.optimizer = hparams.optimizer
        self.momentum = hparams.momentum
        self.dim_input = hparams.dim_input
        self.dim_output = hparams.dim_output
        self.num_neurons = hparams.num_neurons
        # mean/std of normalized_allfeat_train_data2.3e+07.pkl
        self.mean_dox = -1.27951997867828703514925966469757689925046406642650254070758819580078125000000000e-09
        self.std_dox = 7.69664139835670136293005439385994659318157573579810559749603271484375000000000000e-08

        self.example_input_array = {'x': torch.rand(1, hparams.dim_input)}
        
        # =================== QRes Neural Net Architecture ====================
        def layer_block(in_feat, out_feat, is_first=False, is_last=False, normalize=True, dropout=False):
            layer = []
            # =================== Batchnorm ====================
            if normalize:
                layer.append(nn.BatchNorm1d(in_feat))
            # =================== Dropout ====================
            if dropout:
                layer.append(nn.Dropout(p=hparams.dropout_pc))
            # =================== Linear ====================
            layer.append(nn.QResLayer(in_feat, out_feat))
            # =================== Non linear ====================
            if not is_last:
                layer.append(nn.Tanh())
            return layer

        layers = []
        for ind in range(hparams.num_layers):
            is_first = ind == 0
            is_last = ind == hparams.num_layers-1
            dim_in = self.dim_input if is_first else self.num_neurons
            dim_out = self.dim_output if is_last else self.num_neurons
            layers.extend(layer_block(dim_in, dim_out, 
                                      is_first, is_last, 
                                      normalize=hparams.batchnorm, 
                                      dropout=hparams.dropout))

        self.model = nn.Sequential(*layers)
                
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.model(x)
        return x

    def training_step(self, batch, batch_idx):
        # 1. Forward pass:
        x, y = batch
        y = y.view(y.size(0), -1)
        y_hat = self.forward(x)

        # 2. Compute loss
        loss = F.mse_loss(y_hat, y)
        #self.log('train_loss', loss, prog_bar=False, logger=True)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        # 1. Forward pass:
        x, y = batch
        y = y.view(y.size(0), -1)
        y_hat = self.forward(x)

        # 2. Compute loss
        loss = F.mse_loss(y_hat, y)
        return {'val_loss': loss}

    def test_step(self, batch, batch_idx):
        # 1. Forward pass:
        x, y = batch
        y = y.view(y.size(0), -1)
        y_hat = self.forward(x)
        
        # 2. Compute loss
        return {'test_loss': F.mse_loss(y_hat, y)}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs])
        self.log('mean_train_loss', avg_loss.mean(), prog_bar=True, logger=True)
        self.log('std_train_loss', avg_loss.std(), prog_bar=True, logger=True)
        self.log('max_train_loss', avg_loss.max(), prog_bar=True, logger=True)

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs])
        self.log('mean_val_loss', avg_loss.mean(), prog_bar=True, logger=True)
        self.log('std_val_loss', avg_loss.std(), prog_bar=True, logger=True)
        self.log('max_val_loss', avg_loss.max(), prog_bar=True, logger=True)

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs])
        self.log('mean_test_loss', avg_loss.mean(), logger=True)
        self.log('std_test_loss', avg_loss.std(), logger=True)
        self.log('max_test_loss', avg_loss.max(), logger=True)
        
    def predict_vmr(self, batch, batch_idx):
        x = batch
        y_hat = self(x)
        y_hat = y_hat * self.std_dox + self.mean_dox
        return {'prediction': y_hat}
        
    def configure_optimizers(self):
        # =================== Optimization methods ====================
        optimizers = {
            'Adam': torch.optim.Adam(
                self.parameters(), 
                lr=self.learning_rate, 
            #    weight_decay=1e-5
            ),
            'AdamW': torch.optim.AdamW(
                self.parameters(), 
                lr=self.learning_rate, 
                betas=(0.9, 0.999), 
                eps=1e-08, 
            #    weight_decay=0.01
            ),
            'LBFGS': torch.optim.LBFGS(
                self.parameters(),
                lr=1, 
                max_iter=20, 
                max_eval=None, 
                tolerance_grad=1e-07, 
                tolerance_change=1e-09, 
                history_size=100),
            'SGD': torch.optim.SGD(
                self.parameters(), 
                lr=self.learning_rate, 
                momentum=self.momentum),
        }
        optimizer = optimizers[self.optimizer]
        print('using {} optimizer.'.format(self.optimizer))
        return [optimizer]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--dim_input', type=int, default=11)
        parser.add_argument('--dim_output', type=int, default=1)
        parser.add_argument('--num_layers', type=int, default=7)
        parser.add_argument('--num_neurons', type=int, default=250)
        parser.add_argument('--num_traindata', type=int, default=-1)
        parser.add_argument('--num_validata', type=int, default=1000000)
        parser.add_argument('--num_testdata', type=int, default=-1)
        parser.add_argument('--batch_size', type=int, default=100)
        parser.add_argument('--learning_rate', type=float, default=0.00001)
        parser.add_argument('--dropout_pc', type=float, default=0.2)
        parser.add_argument('--dropout', type=bool, default=False)
        parser.add_argument('--batchnorm', type=bool, default=True)
        #parser.add_argument('--num_workers', type=int, default=12)# JUWELS: Dual Intel Xeon Gold 6148 --> 20 cores, 2.4 GH
        return parser