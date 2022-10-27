import os
from datetime import datetime
import logging
import random
import numpy as np
import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.distributed as dist

from mico.model import MutualInfoCotrain, train_mico
from mico.evaluate import infer_on_test
from mico.utils import setup_primary_logging, setup_worker_logging, get_model_specific_argparser

torch.backends.cudnn.benchmark = True


def run_experiment_worker(subprocess_index, world_size, model_mico, hparams, log_queue):
    """This is the function in each sub-process for multi-process training with DistributedDataParallel.

    Parameters
    ----------
    subprocess_index : int
        The index for the current sub-process
    world_size : int
        Total number of all the sub-processes
    model_mico : `MutualInfoCotrain` object
        The same initialized model for all the sub-processes
    hparams : dictionary
        The hyper-parameters for MICO
    log_queue : `torch.multiprocessing.Queue`
        For the logging with multi-process

    """
    setup_worker_logging(subprocess_index, log_queue)
    logging.info("Running experiment on GPU %d" % subprocess_index)
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=subprocess_index, world_size=world_size)

    # For reproducible random runs
    random.seed(hparams.seed * world_size + subprocess_index)
    np.random.seed(hparams.seed * world_size + subprocess_index)
    torch.manual_seed(hparams.seed * world_size + subprocess_index)

    train_mico(subprocess_index, model_mico)


if __name__ == '__main__':
    torch.multiprocessing.set_start_method("spawn", force=True)
    argparser = get_model_specific_argparser()
    hparams = argparser.parse_args()

    os.makedirs(hparams.model_path, exist_ok=True)

    # Model training part
    if not hparams.eval_only: 
        logging_path = hparams.model_path + '/train.' + f"{datetime.now():%Y-%m-%d-%H-%M-%S}" + '.log'
        log_queue, listener = setup_primary_logging(logging_path)
        setup_worker_logging(-1, log_queue)

        print_hparams = ""
        for key in hparams.__dict__:
            if str(hparams.__dict__[key]) == 'False':
                continue
            elif str(hparams.__dict__[key]) == 'True':
                print_hparams += '--{:s} \\\n'.format(key)
            else:
                print_hparams += '--{:s}={:s} \\\n'.format(key, str(hparams.__dict__[key]))
        logging.info("\n=========== Hyperparameters =========== \n" +
                    print_hparams +
                    "\n=======================================")

        world_size = torch.cuda.device_count()
        logging.info("In this machine, we have %d GPU cards." % world_size)

        model_mico = MutualInfoCotrain(hparams) # set it here to ensure they are initialized at the same parameter.
        mp.spawn(run_experiment_worker, args=(world_size, model_mico, hparams, log_queue,), nprocs=world_size, join=True)
        listener.stop()

    # Model testing part
    logging_path = hparams.model_path + '/eval.' + f"{datetime.now():%Y-%m-%d-%H-%M-%S}" + '.log'
    log_queue, listener = setup_primary_logging(logging_path)
    setup_worker_logging(-1, log_queue)

    logging.info("Start testing")
    device = 'cuda' if hparams.cuda else 'cpu'
    model_mico = MutualInfoCotrain(hparams)
    try:
        model_mico.load(suffix='/model_best.pt', subprocess_index=0)
        logging.info("Load and test the best model during training... ")
    except:
        model_mico.load(suffix='/model_current_iter.pt', subprocess_index=0)
        logging.info("Load and test the model of the most recent iteration during training... ")
    model_mico = model_mico.to(device)
    model_mico.model_bert = nn.DataParallel(model_mico.model_bert)
    model_mico.hparams = hparams
    model_mico.hparams.batch_size_test = hparams.batch_size_test * torch.cuda.device_count()
    infer_on_test(model_mico, device)
    listener.stop()
