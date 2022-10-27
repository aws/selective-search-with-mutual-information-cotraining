import torch
from tqdm import tqdm
import numpy as np
import json
import logging
import os
from timeit import default_timer as timer
from datetime import timedelta

from mico.dataloader import QueryDocumentsPair
from mico.utils import monitor_metrics

def evaluate(model, eval_loader, num_batches=-1, device=None):
    """This function evaluates the model during training.

    Parameters
    ----------
    model : MutualInfoCotrain object
        This is the MICO model we have been training.
    eval_loader : PyTorch Dataloader object
        This is the dataloader used for evaluation. It will be val_dataloader and test_dataloader.
    num_batches : int
        If this is set to be a non-negative number, 
        the evaluation will only be done with the first n batches in the dataloader where n = num_batches.
    device : int (for multi-GPU) or string ('cpu' or 'cuda')
        This is the GPU index since we use DistributedDataParallel,
        and the index of the GPU card is also the index of the sub-process.
    
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
    model.eval()
    with torch.no_grad():
        encoding_chunks_query = []
        encoding_chunks_document = []
        for batch_idx, (query, document) in enumerate(tqdm(eval_loader)):
            if batch_idx == num_batches:
                break
            query_prob = model.forward(query=query, forward_method="encode_query", device=device)
            document_prob = model.forward(document=document, forward_method="encode_doc", device=device)
            encoding_chunks_query.append(query_prob)
            encoding_chunks_document.append(document_prob)
        encodings_query = torch.cat(encoding_chunks_query, 0).float()
        encodings_document = torch.cat(encoding_chunks_document, 0).float()
        perf = monitor_metrics(encodings_query, encodings_document)
    model.train()
    return perf


def infer_on_test(model, device=None):
    """This function will assign the documents in all the three datasets (train/val/test), which is saved in `clustered_docs.json`,
        and calculate the metrics for the performance on testing data, which is saved in `metrics.json`.

        The metrics are the coverage for impression, click, purchase; the latency cost and total search cost; and the cluster sizes.

    Parameters
    ----------
    model : MutualInfoCotrain object
        This is the MICO model we have loaded. The path for the train/val/test dataset is in `model.hparams`.
    device : string ('cpu' or 'cuda')
        This is the device we run model on. 

    Returns
    -------
    metrics : 
        This contains the coverage for impression, click, purchase, also the latency cost and total search cost on test dataset.
        This also contains the cluster sizes.
    """
    start = timer()

    hparams = model.hparams
    data = QueryDocumentsPair(train_folder_path=hparams.train_folder_path, test_folder_path=hparams.test_folder_path, 
                              is_csv_header=hparams.is_csv_header, val_ratio=hparams.val_ratio, is_get_all_info=True)
    train_loader, val_loader, test_loader = data.get_loaders(model.hparams.batch_size_test, model.hparams.num_workers,
                                                  is_shuffle_train=False, is_get_test=True)
    model.eval()
    cluster_docs_path = hparams.model_path + '/clustered_docs.json' 
    metrics_save_path = hparams.model_path + '/metrics.json' 
    cluster_docs = {}
    n_clusters = hparams.number_clusters
    for i in range(n_clusters):
        cluster_docs[i] = {}

    with torch.no_grad():
        if os.path.isfile(cluster_docs_path):
            # documents are already clustered for this model.
            logging.info("Loading the existing document clustering results.")
            logging.info("If you want to re-cluster the documents, please delete %s" % cluster_docs_path)
            with open(cluster_docs_path, "r+") as f:
                cluster_docs = json.load(f)
            cluster_docs2 = {}
            for i in cluster_docs:
                cluster_docs2[int(i)] = cluster_docs[i]
            cluster_docs = cluster_docs2
        else:
            # cluster all the documents
            for (query, asin, document, click, purchase) in tqdm(train_loader):
                document_prob = model.forward(document=document, forward_method="encode_doc", device=device)
                for idx2, class_idx in enumerate(torch.argmax(document_prob, dim=1).cpu().numpy()):
                    cluster_docs[class_idx][asin[idx2]] = 1
            for (query, asin, document, click, purchase) in tqdm(val_loader):
                document_prob = model.forward(document=document, forward_method="encode_doc", device=device)
                for idx2, class_idx in enumerate(torch.argmax(document_prob, dim=1).cpu().numpy()):
                    cluster_docs[class_idx][asin[idx2]] = 1
            for (query, asin, document, click, purchase) in tqdm(test_loader):
                document_prob = model.forward(document=document, forward_method="encode_doc", device=device)
                for idx2, class_idx in enumerate(torch.argmax(document_prob, dim=1).cpu().numpy()):
                    cluster_docs[class_idx][asin[idx2]] = 1
            
            logging.info('Save the clusters of all the documents ...')
            with open(cluster_docs_path, 'w+') as f:
                json.dump(cluster_docs, f, indent='\t')
    
        logging.info('Calculate the metrics (coverage for impression, click, and purchase; latency cost, and total search cost) on test dataset ...')
        cluster_sizes = [len(cluster_docs[i]) for i in range(n_clusters)]
        impression_coverage = np.zeros(n_clusters)
        click_coverage = np.zeros(n_clusters)
        purchase_coverage = np.zeros(n_clusters)
        latency_cost_curve = np.zeros(n_clusters)
        total_cost_curve = np.zeros(n_clusters)

        for (query_batch, asin_batch, document_batch, click_batch, purchase_batch) in tqdm(test_loader):
            query_batch_prob = model.forward(query=query_batch, forward_method="encode_query", device=device)
            query_batch_prob = -query_batch_prob.detach().cpu().numpy()
            query_batch_cluster_rank = query_batch_prob.argsort(axis=1) 

            for query_idx in range(len(query_batch)): # this is a batch

                impression_find = np.zeros(n_clusters)
                click_find = np.zeros(n_clusters)
                purchase_find = np.zeros(n_clusters)
                total_time_find = np.zeros(n_clusters)
                latency_find = np.zeros(n_clusters)

                query_one = query_batch[query_idx]
                asin_one = asin_batch[query_idx]
                click_one = click_batch[query_idx]
                purchase_one = purchase_batch[query_idx]
                for rank_idx, cluster_idx in enumerate(query_batch_cluster_rank[query_idx]):
                    total_time_find[rank_idx] = cluster_sizes[cluster_idx]
                    latency_find[rank_idx] = max(latency_find[rank_idx-1], cluster_sizes[cluster_idx]) \
                                                if rank_idx > 0 else cluster_sizes[cluster_idx]
                    if asin_one in cluster_docs[cluster_idx]:
                        impression_find[rank_idx] = 1
                        click_find[rank_idx] = click_one
                        purchase_find[rank_idx] = purchase_one
                        found_asin = True
                if not found_asin:
                    logging.info('ERROR: ASIN %s not found in all clusters when consider query %s' % (asin_one, query_one))
                    raise ValueError

                impression_coverage += np.cumsum(impression_find)
                click_coverage += np.cumsum(click_find)
                purchase_coverage += np.cumsum(purchase_find)
                total_cost_curve += np.cumsum(total_time_find)
                latency_cost_curve += latency_find

        impression_coverage /= impression_coverage[-1]
        click_coverage /= click_coverage[-1]
        purchase_coverage /= purchase_coverage[-1]
        total_cost_curve = total_cost_curve / len(data.test_dataset)
        latency_cost_curve = latency_cost_curve / len(data.test_dataset)
        metrics = { 'impression_coverage': impression_coverage.tolist(), \
                    'click_coverage': click_coverage.tolist(), \
                    'purchase_coverage': purchase_coverage.tolist(), \
                    'total_cost_curve': total_cost_curve.tolist(), \
                    'latency_cost_curve': latency_cost_curve.tolist(), \
                    'cluster_sizes': sorted(cluster_sizes), \
                  }

        logging.info('Save the metrics ...')
        with open(metrics_save_path, 'w+') as f:
            json.dump(metrics, f, indent='\t')

    logging.info("Total testing time: %s" % str(timedelta(seconds=round(timer() - start))))
    return metrics

