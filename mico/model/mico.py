import torch
import torch.nn as nn
from transformers import BertTokenizerFast, BertModel, BertConfig

from mico.utils import ConditionalDistributionZ, MarginalDistributionZ, get_init_function, cross_entropy_p_q
from mico.utils import monitor_metrics, model_predict


class MutualInfoCotrain(nn.Module):
    """MICO method including parameterization, model loading, and model forwarding.

    Here we use BERT as embedding method. 
    When we train MICO, by default we also finetune BERT. 
    If you do not want to finetune BERT, please set `--bert_fix`. **This will degrade the performance a lot.**

    """

    def __init__(self, hparams):
        """We initialize all the parameters for 
                the document assignment model P(Z|Y) which is `self.p_z_y`, where Z is the cluster index to which a document Y is assigned.
                the query routing model P(Z|X) which is `self.q_z_x`, where Z is the cluster index to which a query X is routed.
                the model for approximating the distribution of cluster sizes of document assignment E_Y[P(Z|Y)] which is `self.q_z`, 
                (For details, please check https://arxiv.org/pdf/2209.04378.pdf)

        We load the BERT model and its corresponding tokenizer.


        Parameters
        ----------
        hparams : argparse result
            All the hyper-parameters for MICO. This will also be used in training.

        """
        super().__init__()
        self.hparams = hparams
        self.p_z_y = ConditionalDistributionZ(self.hparams.number_clusters,
                                              self.hparams.dim_input,
                                              self.hparams.num_layers_posterior,
                                              self.hparams.dim_hidden)
        self.q_z = MarginalDistributionZ(self.hparams.number_clusters)
        self.q_z_x = ConditionalDistributionZ(self.hparams.number_clusters,
                                              self.hparams.dim_input,
                                              self.hparams.num_layers_posterior,
                                              self.hparams.dim_hidden)

        self.apply(get_init_function(self.hparams.init))

        try:
            tokenizer = BertTokenizerFast.from_pretrained('bert-base-multilingual-cased', local_files_only=True)
            config_bert = BertConfig.from_pretrained("bert-base-multilingual-cased", output_hidden_states=True,
                                                     local_files_only=True)
            model_bert = BertModel.from_pretrained("bert-base-multilingual-cased", config=config_bert,
                                                   local_files_only=True)
        except:
            # Connect to Internet to download the Huggingface tokenizer, config, and model.
            tokenizer = BertTokenizerFast.from_pretrained('bert-base-multilingual-cased')
            config_bert = BertConfig.from_pretrained("bert-base-multilingual-cased", output_hidden_states=True)
            model_bert = BertModel.from_pretrained("bert-base-multilingual-cased", config=config_bert)

        self.model_bert = model_bert
        self.tokenizer = tokenizer

    def load(self, suffix="", subprocess_index=0):
        """Load the existing model (previously saved on disk).
        If there is optimizer infomation (e.g., for Adam), we also load it for resuming training.

        Parameters
        ----------
        suffix : string
            The model path for loading will be `self.hparams.model_path + suffix`.
            Different models will be loaded if you set different `suffix`.

        """

        checkpoint = torch.load(self.hparams.model_path + suffix,
                                map_location=torch.device(subprocess_index)) \
            if self.hparams.cuda \
            else torch.load(self.hparams.model_path + suffix,
                            map_location=torch.device('cpu'))
        if checkpoint['hparams'].cuda and not self.hparams.cuda:
            checkpoint['hparams'].cuda = False
        if not self.hparams.not_resume_hparams:
            self.__init__(checkpoint['hparams'])
        self.resume_epoch = checkpoint['epoch']
        self.load_state_dict(checkpoint['state_dict'])
        if 'optimizer' in checkpoint:
            self.optimizers = checkpoint['optimizer']
        if 'current_iteration_number' in checkpoint:
            self.resume_iteration = checkpoint['current_iteration_number']
        else:
            self.resume_iteration = 0

    def forward(self, document=None, query=None, is_monitor_forward=False, forward_method="update_all", device=None):
        """This function handles several types of forward:
            1. for training the approximated distribution `q_z` only  (`update_q_z=True`)
            2. for training the MICO including finetuning BERT  
            3. for document assignment (`encode_doc=True`)
            4. for query routing (`encode_query=True`)

        Parameters
        ----------
        document : string
            A list of raw sentence of the document titles in the samples.
        query : string
            A list of raw sentence of the queries in samples.
        is_monitor_forward : bool
            Calculate more information of the forward pass: {'AUC', 'top1_cov', 'H__Z_Y', 'H__Z_X'}
            Details please refer to the function `monitor_metrics`.
        forward_method : string
            Take values in ['update_all', 'update_q_z', 'encode_doc', 'encode_query']
            The default value is 'update_all'.
            Set this to 'update_q_z' if you only want to update the approximated distribution `q_z`.
            Set this to 'encode_doc' if you only want to perform document assignment.
            Set this to 'encode_query' if you only want to perform query routing.
        device : int (for multi-GPU) or string ('cpu' or 'cuda')
            The device that the BERT model is on.

        Returns
        -------
        metrics : 
            By default, return a dictionary: 
                        {'loss': loss, 'h_z_cond': h_z_cond, 'h_z': h_z}. 
                        The datatypes are all PyTorch Tensor scalars (float).
                If `is_monitor_forward=True`, there are extra keys in the dictionary: 
                        {'AUC': metric_auc * 100, 'top1_cov': acc.item() * 100, 'H__Z_X': enc_x_entropy, 'H__Z_Y': enc_y_entropy}
                        The datatypes are all float.
            If `encode_doc=True` or `encode_query=True`, return a PyTorch Tensor about the probability distribution (for this sample belonging to which cluster).
            If `update_q_z=True`, return the cross-entropy loss for updating q_z.
        """
        if query is None or document is None:
            if forward_method == "encode_doc":
                bert_representation = model_predict(self.model_bert, document,
                                                    self.tokenizer, self.hparams.max_length,
                                                    self.hparams.pooling_strategy,
                                                    self.hparams.selected_layer_idx, device).detach()
                p = (self.p_z_y(bert_representation))
                return p.view(p.shape[0], p.shape[-1])
            elif forward_method == "encode_query":
                bert_representation = model_predict(self.model_bert, query,
                                                    self.tokenizer, self.hparams.max_length,
                                                    self.hparams.pooling_strategy,
                                                    self.hparams.selected_layer_idx, device).detach()
                p = (self.q_z_x(bert_representation))
                return p.view(p.shape[0], p.shape[-1])
            else:
                raise ValueError("Unexpected usage of forward.")
        if self.hparams.bert_fix:
            with torch.no_grad():
                query_bert = model_predict(self.model_bert, query,
                                           self.tokenizer, self.hparams.max_length,
                                           self.hparams.pooling_strategy,
                                           self.hparams.selected_layer_idx, device).detach()
                document_bert = model_predict(self.model_bert, document,
                                              self.tokenizer, self.hparams.max_length,
                                              self.hparams.pooling_strategy,
                                              self.hparams.selected_layer_idx, device).detach()
        else:
            query_bert = model_predict(self.model_bert, query,
                                       self.tokenizer, self.hparams.max_length,
                                       self.hparams.pooling_strategy,
                                       self.hparams.selected_layer_idx, device)
            document_bert = model_predict(self.model_bert, document,
                                          self.tokenizer, self.hparams.max_length,
                                          self.hparams.pooling_strategy,
                                          self.hparams.selected_layer_idx, device)

        p = self.p_z_y(document_bert)

        if forward_method == "update_q_z":
            return cross_entropy_p_q(p.detach(), self.q_z())
        elif forward_method != "update_all":
            raise ValueError("Unexpected usage of forward.")

        q = self.q_z_x(query_bert)
        h_z_cond = cross_entropy_p_q(p, q)
        h_z = cross_entropy_p_q(p, self.q_z())

        loss = h_z_cond - self.hparams.entropy_weight * h_z
        results = {'loss': loss, 'h_z_cond': h_z_cond, 'h_z': h_z}
        if is_monitor_forward:
            results_more = monitor_metrics(p.view(p.shape[0], p.shape[-1]), q.view(q.shape[0], q.shape[-1]))
            results.update(results_more)
        return results
