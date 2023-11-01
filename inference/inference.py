##########################################################################################
# Machine Environment Config

DEBUG_MODE = False
USE_CUDA = not DEBUG_MODE
CUDA_DEVICE_NUM = 0


##########################################################################################
# Path Config

import os
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "..")  # for problem_def
sys.path.insert(0, "../..")  # for utils


##########################################################################################
# import

import logging
from utilsdraw.utils import create_logger, copy_all_src

from TSPInference import TSPTrainer as Trainer


##########################################################################################
# parameters
problem_name=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]
problem_size=[9,10,10,11,12,12,10,12,10,10,10,15,28,34,22,33,35,39,42,45,48,55,59,60,66]
for i in range(25):

    env_params = {
        'problem_size': problem_size[i],
        'pomo_size': problem_size[i],
        'problem_name':problem_name[i],
        'decoder_num': 5,
    }

    model_params = {
        'embedding_dim': 128,
        'sqrt_embedding_dim': 128**(1/2),
        'encoder_layer_num': 3,
        'qkv_dim': 16,
        'head_num': 8,
        'logit_clipping': 10,
        'ff_hidden_dim': 512,
        'eval_type': 'argmax',
        'decoder_num': 5,
    }

    optimizer_params = {
        'optimizer': {
            'lr': 1e-4,
            'weight_decay': 1e-6
        },
        'scheduler': {
            'milestones': [1,],
            'gamma': 0.1
        }
    }


    trainer_params = {
        'use_cuda': USE_CUDA,
        'cuda_device_num': CUDA_DEVICE_NUM,
        'epochs': 101,
        'train_episodes': 100*10000,
        'train_batch_size': 8,
        'logging': {
            'model_save_interval': 10,
            'img_save_interval': 10,
            'log_image_params_1': {
                'json_foldername': 'log_image_style',
                'filename': 'style_tsp_20.json'
            },
            'log_image_params_2': {
                'json_foldername': 'log_image_style',
                'filename': 'style_loss_1.json'
            },
        },
        'model_load': {
            'enable': True,  # enable loading pre-trained model
            'path': './model/MSTSP_train_tsp',  # directory path of pre-trained model and log files saved.
            'epoch': 100,  # epoch version of pre-trained model to laod.

        }
    }

    logger_params = {
        'log_file': {
            'desc': 'MSTSP_train_tsp',
            'filename': 'run_log.txt'
        }
    }

    ##########################################################################################
    # main

    def main():
        if DEBUG_MODE:
            _set_debug_mode()

        create_logger(**logger_params)
        _print_config()

        trainer = Trainer(env_params=env_params,
                        model_params=model_params,
                        optimizer_params=optimizer_params,
                        trainer_params=trainer_params)

        copy_all_src(trainer.result_folder)

        trainer.run()


    def _set_debug_mode():
        global trainer_params
        trainer_params['epochs'] = 2
        trainer_params['train_episodes'] = 10
        trainer_params['train_batch_size'] = 4


    def _print_config():
        logger = logging.getLogger('root')
        logger.info('DEBUG_MODE: {}'.format(DEBUG_MODE))
        logger.info('USE_CUDA: {}, CUDA_DEVICE_NUM: {}'.format(USE_CUDA, CUDA_DEVICE_NUM))
        [logger.info(g_key + "{}".format(globals()[g_key])) for g_key in globals().keys() if g_key.endswith('params')]



    ##########################################################################################

    if __name__ == "__main__":
        main()
