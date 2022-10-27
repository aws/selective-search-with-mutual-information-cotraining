import torch
from sklearn import metrics
import argparse

def assign_gpu(tokenizer_output, device='cuda'):
    """This function helps move the tokenizer output to GPU since we use the BERT model on GPU.
    """
    tokens_tensor = tokenizer_output['input_ids'].to(device)
    token_type_ids = tokenizer_output['token_type_ids'].to(device)
    attention_mask = tokenizer_output['attention_mask'].to(device)

    output = {'input_ids': tokens_tensor,
              'token_type_ids': token_type_ids,
              'attention_mask': attention_mask}
    return output


def model_predict(model, text, tokenizer, max_length, pooling_strategy, selected_layer_idx, device=None):
    """This function generates BERT representation.

    Parameters
    ----------
    model : BertModel
        This is the BERT model.
    text : string
        This is the input text.
    tokenizer : `BertTokenizerFast`
        This is the corresponding tokenizer for the BERT model.
    max_length : int
        This is the maximum of the length of the tokenized sequence.
        If a sentence is too long, the part outside this length will be trucated.
        FYI, the length of the tokenized results is usually larger than the number of words in the sentence. 
    pooling_strategy : string
        This is for the output. 
        If `pooling_strategy=CLS_TOKEN`, we only use the output of the first [CLS] token.
        If `pooling_strategy=REDUCE_MEAN`, we only use the average output of all the tokens.
    selected_layer_idx : int
        This is for the output. We collect the output from which layer. 
        Layer -1 means the last layer. BERT base model has 12 layers.
    device : int (GPU index) or string ('cpu' or 'cuda')
        This is the GPU index that BERT model is on.
    
    Returns
    -------
    BERT representation : `torch.tensor`
        A tensor with 768 dimension (for the BERT base model).
    """
    encoded_input = tokenizer(list(text), return_tensors='pt', padding=True, truncation=True, max_length=max_length)
    encoded_input = assign_gpu(encoded_input, device=device)
    outputs = model(**encoded_input)
    hidden_states_layer = outputs[2][selected_layer_idx]
    if pooling_strategy == 'CLS_TOKEN':
        return hidden_states_layer[:, 0, :]
    elif pooling_strategy == 'REDUCE_MEAN':
        unmasked_indice = (encoded_input['attention_mask'] == 1).float()
        return torch.mul(1 / torch.sum(unmasked_indice, axis=1).unsqueeze(1),
                         torch.einsum('ijk, ij -> ik', hidden_states_layer, unmasked_indice))
    else:
        error_message = "--pooling_strategy=%s is not supported. Please use 'CLS_TOKEN' or 'REDUCE_MEAN'." \
                        % pooling_strategy
        raise NotImplementedError(error_message)


def monitor_metrics(encodings_x, encodings_y):
    """This function helps evaluate the model during training.

    Parameters
    ----------
    encodings_x : `torch.tensor`
        This is the probability of queries routed to each cluster according to MICO model.
    encodings_y : `torch.tensor`
        This is the probability of document assigned to each cluster according to MICO model.
    
    Returns
    -------
    perf : dictionary
        This performance metrics are in a dictionary:
         {'AUC': area under the curve (search cost v.s. top-1 coverage) in percentage;
          'top1_cov': top-1 coverage in percentage,
                      in how much proportion of the data (query-document pair), the top-1 cluster for query routing 
                      and the top-1 cluster for document assignment are the same cluster;
          'H__Z_X': the entropy of cluster distribution of query routing (to their top-1 cluster) in this evaluation data;
          'H__Z_Y': the entropy of cluster distribution of document assigment (to their top-1 cluster) in this evaluation data;
         }
    """
    argmax_x = torch.argmax(encodings_x, dim=1)
    argmax_y = torch.argmax(encodings_y, dim=1)
    num_correct = torch.sum(argmax_x == argmax_y).float()
    acc = num_correct / encodings_x.size(0)
    enc_x_list = []
    enc_y_list = []
    enc_x_prob = []
    enc_y_prob = []
    for i in range(64):
        count_x = torch.sum(argmax_x == i).item()
        count_y = torch.sum(argmax_y == i).item()
        enc_x_list.append(count_x)
        enc_y_list.append(count_y)
        if count_x > 0:
            enc_x_prob.append(count_x)
        if count_y > 0:
            enc_y_prob.append(count_y)
    enc_y_list = torch.tensor(enc_y_list).float()
    enc_x_prob = torch.tensor(enc_x_prob).float()
    enc_y_prob = torch.tensor(enc_y_prob).float()

    enc_x_prob = enc_x_prob / torch.sum(enc_x_prob)
    enc_y_prob = enc_y_prob / torch.sum(enc_y_prob)
    enc_x_entropy = torch.sum(-enc_x_prob * torch.log(enc_x_prob)).item()
    enc_y_entropy = torch.sum(-enc_y_prob * torch.log(enc_y_prob)).item()

    argsort_x = torch.argsort(-encodings_x, dim=1)
    total_cost_mat = torch.cat(
        [torch.cumsum(enc_y_list[argsort_x[i]], 0).unsqueeze(0) for i in range(encodings_x.shape[0])], dim=0)
    total_cost_mean_curve = torch.mean(total_cost_mat, dim=0)
    total_cost_mean_curve = total_cost_mean_curve / total_cost_mean_curve[-1]
    coverage_mat = torch.cat(
        [torch.cumsum((argsort_x[i] == argmax_y[i]).float(), dim=0).unsqueeze(0) for i in range(encodings_x.shape[0])],
        dim=0)
    coverage_curve = torch.mean(coverage_mat, dim=0).cpu()
    metric_auc = metrics.auc(torch.cat([torch.tensor([0]).float(), total_cost_mean_curve], dim=0),
                             torch.cat([torch.tensor([0]).float(), coverage_curve], dim=0))

    return {'AUC': metric_auc * 100, 'top1_cov': acc.item() * 100,
            'H__Z_X': enc_x_entropy, 'H__Z_Y': enc_y_entropy}


def get_model_specific_argparser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_path', type=str)
    parser.add_argument('--train_folder_path', type=str)
    parser.add_argument('--test_folder_path', type=str)
    parser.add_argument('--number_clusters', type=int, default=16,
                        help='number of clusters to which we are assigning documents and routing queries [%(default)d]')
    parser.add_argument('--dim_hidden', type=int, default=24,
                        help='dimension of hidden state [%(default)d]')
    parser.add_argument('--dim_input', type=int, default=24,
                        help='dimension of input [%(default)d]')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='batch size [%(default)d]')
    parser.add_argument('--batch_size_test', type=int, default=16,
                        help='batch size for evaluating the test dataset [%(default)d]')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                        help='proportion of training data used as validation [%(default)g]')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='initial learning rate [%(default)g]')
    parser.add_argument('--init', type=float, default=0.1,
                        help='unif init range (default if 0) [%(default)g]')
    parser.add_argument('--clip', type=float, default=10,
                        help='gradient clipping [%(default)g]')
    parser.add_argument('--epochs', type=int, default=40,
                        help='max number of epochs [%(default)d]')
    parser.add_argument('--save_per_num_epoch', type=int, default=1,
                        help='save model per number of epochs [%(default)d]')
    parser.add_argument('--log_interval', type=int, default=100,
                        help='number of updates for check [%(default)d]')
    parser.add_argument('--check_val_test_interval', type=int, default=10000,
                        help='number of updates for check [%(default)d]')
    parser.add_argument('--num_bad_epochs', type=int, default=10000,
                        help='num indulged bad epochs [%(default)d]')
    parser.add_argument('--early_quit', type=int, default=0,
                        help='num batches training before end the epoch [%(default)d]')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='num dataloader workers [%(default)d]')
    parser.add_argument('--seed', type=int, default=9061,
                        help='random seed [%(default)d]')
    parser.add_argument('--resume', action='store_true',
                        help='Resume the training process.')
    parser.add_argument('--not_resume_hparams', action='store_true',
                        help='Resume the training process and use newly assigned hyper-parameters.')
    parser.add_argument('--cuda', action='store_true',
                        help='use CUDA?')
    parser.add_argument('--eval_only', action='store_true',
                        help='We only evaluate the trained model.')
    parser.add_argument('--bert_fix', action='store_true',
                        help='Finetune BERT or not.')
    parser.add_argument('--is_csv_header', action='store_true',
                        help='If the input CSV files has header (so we skip the first line).')

    parser.add_argument('--num_layers_posterior', type=int, default=0,
                        help='num layers in posterior [%(default)d]')
    parser.add_argument('--num_steps_prior', type=int, default=4,
                        help='num gradient steps on prior per loss '
                                '[%(default)d]')
    parser.add_argument('--lr_prior', type=float, default=0.1,
                        help='initial learning rate for prior (same as lr '
                                ' if -1) [%(default)g]')
    parser.add_argument('--entropy_weight', type=float, default=2,
                        help='entropy weight in MI [%(default)g]')

    # BERT related
    parser.add_argument('--lr_bert', type=float, default=-1,
                        help='initial learning rate for BERT (same as lr '
                                ' if -1) [%(default)g]')
    parser.add_argument('--max_length', type=int, default=4,
                        help='max number of tokens in a sentence for BERT input '
                                '[%(default)d]')
    parser.add_argument('--num_warmup_steps', type=int, default=0,
                        help='warmup steps (linearly increase learning rate)'
                                '[%(default)d]')
    parser.add_argument('--pooling_strategy', type=str, default='REDUCE_MEAN',
                        help='REDUCE_MEAN or CLS_TOKEN [%(default)s]')
    parser.add_argument('--selected_layer_idx', type=int, default=-1,
                        help='output from which layer [%(default)d]')

    return parser

