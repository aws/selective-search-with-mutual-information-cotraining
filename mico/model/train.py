from timeit import default_timer as timer
from datetime import timedelta
import os
import torch
import torch.nn as nn
import math
from torch.nn.parallel import DistributedDataParallel
import logging
from tensorboardX import SummaryWriter
from transformers import get_constant_schedule_with_warmup

from mico.evaluate import evaluate
from mico.dataloader import QueryDocumentsPair


def train_mico(subprocess_index, model):
    """This is the main training function.

    There is an inner loop for updating `q_z`.
    The outer loop is both optimizing MICO and finetuning BERT.

    Parameters
    ----------
    subprocess_index : int
        The index of the sub-process. 
    model : MutualInfoCotrain object
        This is the MICO model we are going to train.

    """
    start = timer()

    hparams = model.hparams
    data = QueryDocumentsPair(train_folder_path=hparams.train_folder_path, test_folder_path=hparams.test_folder_path, 
                              is_csv_header=hparams.is_csv_header, val_ratio=hparams.val_ratio)
    if subprocess_index == 0: # The logging only happens in the first sub-process.
        logging.info("Number of queries in each dataset:    Train / Val / Test = %d / %d / %d" %
                     (len(data.train_dataset), len(data.val_dataset), len(data.test_dataset)))

    starting_epoch = 1
    resume = hparams.resume
    if resume:
        load_path_suffix = '/model_current_iter.pt'
        load_path = hparams.model_path + load_path_suffix
        if not os.path.isfile(load_path):
            load_path_suffix = '/model_current_epoch.pt'
            load_path = hparams.model_path + load_path_suffix
            if not os.path.isfile(load_path):
                logging.info("No previous training model file found in the path %s" % load_path)
                raise ValueError("Although you set --resume, there is no model that we can load.")

        model.load(suffix=load_path_suffix, subprocess_index=subprocess_index) # Here you can also set it to load the model of the most recent epoch.
        model.to(subprocess_index)

        resume_iteration = model.resume_iteration
        starting_epoch = model.resume_epoch
        logging.info('Resume training from iteration %d' % resume_iteration)
        hparams = model.hparams
    else:
        resume_iteration = 0
        model.to(subprocess_index)

    # find_unused_parameters=True is needed since when we update `q_z`, 
    # there are lots of other parameters in MICO and BERT do not have gradients and are not changed.
    model = DistributedDataParallel(model, device_ids=[subprocess_index], find_unused_parameters=True)
    if subprocess_index == 0:
        logging.info("{} params in this whole model.".format(sum([p.numel() for p in model.parameters()])))
        tb_logging = SummaryWriter(hparams.model_path + '/log')

    # initialize optimizers
    mico_param_list = []
    bert_param_list = []
    mico_q_z_param_list = []
    for name, param in (model.named_parameters()):
        if 'q_z' in name:
            mico_q_z_param_list.append(param)
        if 'p_z_y' in name or 'q_z' in name:
            mico_param_list.append(param)
        elif 'bert' in name:
            bert_param_list.append(param)
        else:
            raise ValueError("%s is not in MICO or in BERT. What is this parameter?" % name)

    params = [{"name": "MICO", "params": mico_param_list}]
    if not hparams.bert_fix:
        params.append({"name": "BERT", "params": bert_param_list, "lr": hparams.lr_bert})
    optimizer = torch.optim.Adam(params, lr=hparams.lr)
    optimizers = [optimizer]
    lr_schedulers = []
    for optimizer in optimizers:
        lr_schedulers.append(get_constant_schedule_with_warmup(optimizer=optimizer,
                                                               num_warmup_steps=hparams.num_warmup_steps))

    gradient_clippers = [(mico_param_list, hparams.clip)]
    if not hparams.bert_fix:
        gradient_clippers.append((bert_param_list, hparams.clip))

    # here we obtain test_loader to check whether the performance on the test set
    # is similar to the one on validation set. 
    # We do not make any decision by test performance. 
    # The best model is selected via the validation performance.
    train_loader, val_loader, test_loader = data.get_loaders(
        hparams.batch_size, hparams.num_workers,
        is_shuffle_train=True, is_get_test=True)
    best_val_perf = float('-inf')
    forward_sum = {'loss': 0}
    num_steps = 0
    bad_epochs = 0

    try:
        for epoch in (range(starting_epoch, hparams.epochs + 1)):
            model.train()
            start = timer()
            if epoch <= resume_iteration / len(train_loader):
                continue
            elif epoch == (resume_iteration // len(train_loader) + 1):
                starting_iteration = resume_iteration % len(train_loader) + 1
            else:
                starting_iteration = 0
            for batch_num, batch in enumerate(train_loader):
                batch_num += starting_iteration
                if batch_num >= len(train_loader):
                    break
                current_iteration_number = (epoch - 1) * len(train_loader) + batch_num
                if hparams.early_quit and hparams.early_quit == batch_num:
                    logging.info('stop early at hparams.early_quit')
                    break
                for optimizer in optimizers:
                    optimizer.zero_grad()
                query, document = batch

                if batch_num % hparams.log_interval == 0:
                    is_monitor_forward = True
                else:
                    is_monitor_forward = False

                # Here we only optimize `q_z` for a better evaluation of later cross entropy.
                q_z_loss = model.forward(document=document, query=query, is_monitor_forward=is_monitor_forward, forward_method="update_q_z", device=subprocess_index)

                optimizer_prior = torch.optim.Adam(mico_q_z_param_list, lr=hparams.lr_prior)
                for _ in range(hparams.num_steps_prior):
                    optimizer_prior.zero_grad()
                    q_z_loss.backward()
                    nn.utils.clip_grad_norm_(mico_q_z_param_list, hparams.clip)
                    optimizer_prior.step()

                # Now we can evaluate the cross entropy between query routing and document assignment
                forward = model.forward(document=document, query=query, is_monitor_forward=is_monitor_forward, forward_method="update_all", device=subprocess_index)

                forward['loss'].backward()

                for params, clip in gradient_clippers:
                    nn.utils.clip_grad_norm_(params, clip)

                for optimizer, lr_scheduler in zip(optimizers, lr_schedulers):
                    optimizer.step()
                    lr_scheduler.step()

                if subprocess_index == 0: # This part is only for monitoring the training process and writing logs.
                    for key in forward:
                        if key in forward_sum:
                            try:
                                forward_sum[key] += forward[key].detach().item()
                            except:
                                forward_sum[key] += forward[key]
                        else:
                            try:
                                forward_sum[key] = forward[key].detach().item()
                            except:
                                forward_sum[key] = forward[key]
                    num_steps += 1

                    if batch_num % (20 * hparams.log_interval) == 0:
                        logging.info('Epoch\t | \t batch \t | \tlr_MICO\t | \tlr_BERT\t' + \
                                     '\t'.join([' | {:8s}'.format(key) \
                                                for key in forward_sum]))

                    for param_group in optimizer.param_groups:
                        if param_group['name'] == 'MICO':
                            curr_lr_mico = param_group['lr']
                        if param_group['name'] == 'BERT':
                            curr_lr_bert = param_group['lr']

                    if batch_num % hparams.log_interval == 0:
                        logging.info('{:d}\t | {:5d}/{:5d} \t | {:.3e} \t | {:.3e} \t'.format(
                            epoch, batch_num, len(train_loader), curr_lr_mico, curr_lr_bert) +
                                     '\t'.join([' | {:8.2f}'.format(forward[key])
                                                for key in forward_sum]))
                        for key in forward_sum:
                            tb_logging.add_scalar('Train/' + key, forward[key], current_iteration_number)

                    if current_iteration_number >= 0 and current_iteration_number % hparams.check_val_test_interval == 0:
                        part_batch_num = 1000
                        val_part_perf = evaluate(model, val_loader, num_batches=part_batch_num, device=subprocess_index)
                        test_part_perf = evaluate(model, test_loader, num_batches=part_batch_num, device=subprocess_index)

                        logging.info('Evaluate on {:d} batches | '.format(part_batch_num) +
                                     'Iteration number {:10d}'.format(current_iteration_number) +
                                     ' '.join([' | val_partial {:s} {:8.2f}'.format(key, val_part_perf[key])
                                               for key in val_part_perf]) +
                                     ' '.join([' | test_partial {:s} {:8.2f}'.format(key, test_part_perf[key])
                                               for key in test_part_perf]))

                        for key in val_part_perf:
                            tb_logging.add_scalar('Val_partial/' + key, val_part_perf[key], current_iteration_number)
                        for key in test_part_perf:
                            tb_logging.add_scalar('Test_partial/' + key, test_part_perf[key], current_iteration_number)

                        torch.save(
                            {'hparams': hparams, 'state_dict': model.module.state_dict(), 'optimizer': optimizers,
                             'epoch': epoch, 'current_iteration_number': current_iteration_number},
                            hparams.model_path + '/model_current_iter.pt')

                    if math.isnan(forward_sum['loss']):
                        logging.info("Stopping epoch because loss is NaN")
                        break

                    tb_logging.flush()

            if math.isnan(forward_sum['loss']):
                logging.info("Stopping training session at ep %d batch %d because loss is NaN" % (epoch, batch_num))
                break

            if subprocess_index == 0:
                val_perf = evaluate(model, val_loader, num_batches=10000, device=subprocess_index)
                test_perf = evaluate(model, test_loader, num_batches=10000, device=subprocess_index)

                logging.info('End of epoch {:3d}'.format(epoch) +
                             ' '.join([' | train ave {:s} {:8.2f}'.format(key, forward_sum[key] / num_steps)
                                       for key in ['loss', 'h_z_cond', 'h_z']]) +
                             ' '.join([' | val {:s} {:8.2f}'.format(key, val_perf[key])
                                       for key in val_perf]) +
                             ' '.join([' | test {:s} {:8.2f}'.format(key, test_perf[key])
                                       for key in test_perf]) +
                             ' | Time %s' % str(timedelta(seconds=round(timer() - start))))

                for key in ['loss', 'h_z_cond', 'h_z']:
                    tb_logging.add_scalar('Train/ave_' + key, forward_sum[key] / num_steps, epoch)
                for key in val_perf:
                    tb_logging.add_scalar('Val/' + key, val_perf[key], epoch)
                for key in test_perf:
                    tb_logging.add_scalar('Test/' + key, test_perf[key], epoch)
                for key in val_perf:
                    tb_logging.add_scalar('Val_in_steps/' + key, val_perf[key], current_iteration_number)
                for key in test_perf:
                    tb_logging.add_scalar('Test_in_steps/' + key, test_perf[key], current_iteration_number)

                forward_sum = {}
                num_steps = 0

                val_perf = val_perf['AUC']
                if val_perf > best_val_perf:
                    best_val_perf = val_perf
                    bad_epochs = 0
                    logging.info('*** Best model so far, saving ***')
                    torch.save({'hparams': hparams, 'state_dict': model.module.state_dict(), 'epoch': epoch},
                               hparams.model_path + '/model_best.pt')
                else:
                    bad_epochs += 1
                    logging.info("Bad epoch %d" % bad_epochs)

                torch.save({'hparams': hparams, 'state_dict': model.module.state_dict(), 'optimizer': optimizers,
                            'epoch': epoch, 'current_iteration_number': current_iteration_number},
                           hparams.model_path + '/model_current_epoch.pt')

                if bad_epochs > hparams.num_bad_epochs:
                    break

                if epoch % hparams.save_per_num_epoch == 0:
                    torch.save({'hparams': hparams, 'state_dict': model.module.state_dict(), 'optimizer': optimizers,
                                'epoch': epoch, 'current_iteration_number': current_iteration_number},
                               hparams.model_path + '/model_opt_epoch' + str(epoch) + '.pt')

                tb_logging.flush()

    except KeyboardInterrupt:
        if subprocess_index == 0:
            logging.info('-' * 40)
            logging.info('Exiting from training early')

    if subprocess_index == 0:
        tb_logging.close()
        logging.info("Total training time: %s" % str(timedelta(seconds=round(timer() - start))))
