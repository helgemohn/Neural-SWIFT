###############################
# Copyright (c) 2023 Helge Mohn
# SPDX-License-Identifier: MIT License
###############################



###############################


if __name__ == "__main__":
    
    # ------------
    # args
    # ------------
    import os
    os.environ["WANDB_MODE"] = "offline"
    
    from tools.model_args import get_args
    from tools.model_siren import DeepNeuralNet
    from tools.model_dataloader import SwiftAI_DataModule
    hparams = get_args(DeepNeuralNet, SwiftAI_DataModule)
    hparams.project_name = 'Neural-SWIFT'
    hparams.group_name = 'final'
    hparams.version = 1.0

    from pytorch_lightning import seed_everything
    seed_everything(hparams.seed, workers=True)

    variables_all = ["modeldate", "dayofyear", "month", "year", "overhead", "eqlat", "sza", "daylight", "O2_reaction_coef", "O3_reaction_coef", "NOx_reaction_coef", "NOy_reaction_coef", "ClOy_reaction_coef", "ClOx_reaction_coef", "longitude", "latitude", "z", "temperature", "theta", "pv", "Cly", "Bry", "NOy", "HOy", "Ox", "dOx", "dCly", "dBry", "dNOy", "dHOy"]
    variables_in = ['Cly', 'Bry', 'NOy', 'HOy', 'Ox', 'z', 'overhead', 'temperature', "daylight", "O2_reaction_coef", "O3_reaction_coef", "ClOy_reaction_coef", "ClOx_reaction_coef"]
    variable_out = ['dOx']
    
    assert hparams.mode == 'training' or hparams.mode == 'testing', 'Option --testing: Please choose a valid mode from {training, testing}'

    # ------------
    # --month: Neural-SWIFT consists of twelve models - a model for each calendar month (1 .. 12)
    # ------------
    assert hparams.month > 0 and hparams.month < 13, 'Option --month: Please choose a valid number calendar month (1 .. 12).'
    import calendar
    print('*** Neural-SWIFT for month {} ***'.format(calendar.month_name[hparams.month]))
    model_path = "../../models/python/paper1_model_{:02d}.ckpt".format(hparams.month)
    hparams.data_path_train = '../../data/training/SWIFT-AI_train_month_{:02d}.nc'.format(hparams.month)
    hparams.data_path_test = '../../data/testing/SWIFT-AI_test_month_{:02d}.nc'.format(hparams.month)
    
#    import os
#    if os.path.isfile(hparams.data_path_train):
#        # file exists
#        f = open("filename.txt")
    
    if hparams.mode == 'training':
        print('Training-Mode: train a new model using the training data')
        # ------------
        # update args
        # ------------    
        hparams.job_type = "train"
        
        # Limit Model runtime
        hparams.max_epochs = 10
        #hparams.val_check_interval = 1000
        hparams.max_steps = 20000
        
        # update args received by wandb
        import wandb
        with wandb.init() as run:
            config = wandb.config
            #print(config)
            for key in config.keys():
                if key == 'gpus':
                    setattr(hparams, key, [config[key]] )
                elif key == 'features_in':
                    setattr(hparams, key, config[key])
                    features_in = config[key].replace("'", "").replace(",", "").replace("[", "").replace("]", "").split()
                else:
                    setattr(hparams, key, config[key] )

            import json
            wandb.run_name = json.dumps(dict(config))      

        hparams.features_in = str(variables_in)
        hparams.features_out = str(variable_out)
        hparams.dim_input = len(variables_in)
        hparams.dim_output = len(variable_out)
        # ------------
        # data
        # ------------
        data = SwiftAI_DataModule(hparams, variables_in, variable_out)

        # ------------
        # model
        # ------------
        model = DeepNeuralNet(hparams)

        model.mean_dox = data.mean[-1]
        model.std_dox = data.std[-1]

        # ------------
        # ready to start training
        # ------------
        print('The training data is being used to train a new model with the known regression output.\n' +
              'The training data is a tabular data set having rows of input and output variables).\n' +
              'mean_val_loss: Volume Mixing Ratio of ozone (we use the Odd Oxygen Family as a proxy for ozone).')
        from tools.model_training import train
        train(model, data, hparams)
