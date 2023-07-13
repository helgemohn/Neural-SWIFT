from argparse import Namespace
from pytorch_lightning import LightningDataModule
import uuid

def test(model_path: str, data: LightningDataModule, hparams: Namespace) -> None:
    """
    :param hparams: The arguments to run the model with.
    """
    # ------------
    # logger
    # ------------
    from pytorch_lightning.loggers import TestTubeLogger
    import uuid
    logger = TestTubeLogger(
        save_dir='{}/{}'.format(hparams.default_root_dir, 'test_'+str(uuid.uuid4())),
        name=hparams.exp_name,
        description=hparams.exp_description,)
    
    # ------------
    # model
    # ------------
    from tools.model_siren import DeepNeuralNet  
    model = DeepNeuralNet.load_from_checkpoint(model_path, w0=4., w0_initial=15.,)
    model.mean_dox = data.mean[-1]
    model.std_dox = data.std[-1]
    
    for param_tensor in model.model.state_dict():
        print(str(param_tensor).replace('.', '_'), "\t", model.model.state_dict()[param_tensor].size())
    
    from pytorch_lightning import Trainer
    trainer = Trainer(gpus=1, logger=logger,)
    
    # ------------
    # predicting
    # ------------
    print('*** predicting ***')
    result = trainer.predict(model, dataloaders=data)
    loss = []
    for i in range(len(result)):
        loss.extend(result[i]['predict_loss'])
    
    import numpy as np
    loss = np.array(loss)
    result = {'MSE_predict_loss': np.mean(loss**2),
                'RMSE_predict_loss': np.sqrt(np.mean(loss)),
              'std_mse_predict_loss': np.std(np.abs(loss**2)),
              'mean_abs_predict_loss': np.mean(np.abs(loss)),
               'std_abs_predict_loss': np.std(np.abs(loss)),
               'max_abs_predict_loss': np.max(np.abs(loss))}
    
    from pprint import pprint
    for k in result.keys():
        pprint('{}: {:.2e}'.format(k, result[k]))
    
    # ------------
    # testing
    # ------------
    print('*** testing ***')
    result = trainer.test(model)