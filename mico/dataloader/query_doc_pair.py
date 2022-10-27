from torch.utils.data import Dataset, DataLoader, ConcatDataset
import numpy as np
import logging
import os
import csv
import json


class QueryDocumentsPair:
    """An online dataloader which reads CSV with memory efficient caching.

    Please use the `get_loaders` method to obtain the dataloader. 
    For advanced usage, you can obtain the train/val/test dataset by 
        `data.train_dataset`, `data.val_dataset`, `data.test_dataset` when the instance name is `data`.
    """
    def __init__(self, train_folder_path=None, test_folder_path=None, is_csv_header=True, val_ratio=0.1, is_get_all_info=False):
        """We load the training and test datasets here and split the training data into non-overlapping training and validation.
        
        Parameters
        ----------
        train_folder_path : string
            The path to the folder containing CSV files for training and validation.
        test_folder_path : string
            The path to the folder containing CSV files for testing.
        is_csv_header : bool
            When reading CSV files as input, set `True` if they have headers, so we can skip the first line.
        val_ratio : float
            How much of the training data will be in the validation dataset.
            The rest will be put in training dataset.
        is_get_all_info : bool
            Get the id, click, and purchase for final evaluation.

        """
        self._is_csv_header = is_csv_header
        train_files = list(map(lambda x : train_folder_path + '/' + x, (filter(lambda x : x.endswith("csv"), sorted(os.listdir(train_folder_path))))))
        train_dataset_list = []
        val_dataset_list = []
        for csv_file in train_files:
            train_dataset = LazyTextDataset(csv_file, val_ratio=val_ratio, is_csv_header=self._is_csv_header, is_get_all_info=is_get_all_info)
            train_dataset_list.append(train_dataset)
            val_dataset = LazyTextDataset(csv_file, val_indices=train_dataset.val_indices, is_csv_header=self._is_csv_header, is_get_all_info=is_get_all_info)
            val_dataset_list.append(val_dataset)
        self.train_dataset = ConcatDataset(train_dataset_list)
        logging.info('train_dataset sample size: %d' % self.train_dataset.__len__())
        self.val_dataset = ConcatDataset(val_dataset_list)
        logging.info('val_dataset sample size: %d' % self.val_dataset.__len__())

        test_files = list(map(lambda x : test_folder_path + '/' + x, (filter(lambda x : x.endswith("csv"), sorted(os.listdir(test_folder_path))))))
        test_dataset_list = list(map(lambda x : LazyTextDataset(x, is_csv_header=self._is_csv_header, is_get_all_info=is_get_all_info), test_files))
        self.test_dataset = ConcatDataset(test_dataset_list)
        logging.info('test_dataset sample size: %d' % self.test_dataset.__len__())

    def get_loaders(self, batch_size, num_workers, is_shuffle_train=True, is_get_test=True, prefetch_factor=2, pin_memory=False):
        """Get train/val/test loaders for training and testing.

        Note
        ----
        Setting `pin_memory=True` or larger `prefetch_factor` may increase the speed a little, 
            but it costs much more memory.

        Parameters
        ----------
        batch_size : int
            The batch_size is for each process on each GPU.
        num_workers : int
            Setting this larger than 1 means we have more threads for loading text data.
            It may increase the speed a little but cost much more memory.
        is_shuffle_train : bool
            Whether we shuffle the training data.
        is_get_test : bool
            If you want to make sure the test set is not used, set it to be `False`.

        Returns
        -------
        (train_loader, val_loader, test_loader) : 
            The three dataloaders are PyTorch Dataloader objects.

        """
        train_loader = DataLoader(self.train_dataset, batch_size=batch_size,
                                  num_workers=num_workers, pin_memory=pin_memory,
                                  shuffle=is_shuffle_train,
                                  prefetch_factor=prefetch_factor)
        val_loader = DataLoader(self.val_dataset, batch_size=batch_size,
                                num_workers=num_workers, shuffle=True, pin_memory=pin_memory,
                                prefetch_factor=prefetch_factor)
        test_loader = DataLoader(self.test_dataset, batch_size=batch_size,
                                 num_workers=num_workers, shuffle=True, pin_memory=pin_memory,
                                 prefetch_factor=prefetch_factor) if is_get_test else None
        return train_loader, val_loader, test_loader

def generate_json_for_csv_line_offset(csv_filepath, save_per_line=1):
    """This is a function for caching the offset, so we can efficient find a random line in the CSV file.
    We save the offset data using `json` format on disk and `np.array` format in memory.
    The first number in the `json` file is the total line number of the CSV. The rest numbers are offsets.

    On the disk, if a CSV file is named `example.csv`, the offset file will be saved in the same folder,
        and named as `example.csv.offset_per_*.json` where `*` is the `save_per_line` parameter.

    Parameters
    ----------
    csv_filepath : string
        The path to the CSV file which we are calculating the offset on. 
    save_per_line : int
        If it is larger than 1, we only save the offset per `save_per_line` line. 
        This will increase the reading time a little, but save a lot of memory usage for the offset.

    Returns
    -------
    offset_idx : int
        This is the total line number of the CSV which is also the sample size of this file.
    offset_data : numpy.array
        This is the offset data.
    """
    if not os.path.isfile(csv_filepath):
        raise ValueError("CSV File %s does not exist" % csv_filepath)
    offset_json_filepath = csv_filepath + '.offset_per' + str(save_per_line) + '.json'
    if os.path.isfile(offset_json_filepath):
        # Offset files already exists, just read it
        with open(offset_json_filepath, 'r') as f:
            offset_data = json.load(f) 
        return offset_data[0], np.array(offset_data[1:])
    offset_loc = 0
    offset_idx = 0
    offset_data = [offset_loc]
    with open(csv_filepath, 'rb') as csv_file:
        for line in csv_file:
            offset_idx += 1
            offset_loc += len(line)
            if offset_idx % save_per_line == 0:
                offset_data.append(offset_loc)
    
    offset_data = [offset_idx] + offset_data
    with open(offset_json_filepath, 'w+') as f:
        json.dump(offset_data, f, indent='\t')
    return offset_idx, np.array(offset_data[1:])

class LazyTextDataset(Dataset):
    """An dataset which loads CSV with memory efficient caching.

    Here we use the offsets for each line in the CSV files to speed up random fetching a line.
    For splitting training and validation data, we use block-wise split which saves some memory.
    """
    def __init__(self, filename, val_ratio=0, val_indices=None, is_csv_header=True, is_get_all_info=False):
        """We generate or load the offset data of the CSV file (if it exists). 
        If we are generating training or validation data, we use blocks to split them and create a mapping 
            for reading the raw CSV rows as training or validation samples.
        
        Parameters
        ----------
        filename : string
            The path to the CSV file.
            It should be in the format [query, ID, doc, click, purchase]
        val_ratio : float
            How much of the training data will be in the validation dataset.
            The rest will be put in training dataset.
            If this is set, we know that we are generating the training dataset instead of validation or testing.
        val_indices : numpy.array
            Which sample in the data is used as validation data. 
            If this is set, we know that we are generating the validation dataset instead of training or testing.
        is_csv_header : bool
            When reading CSV files as input, set `True` if they have headers, so we can skip the first line.
        is_get_all_info : bool
            Set this to be True if we are running evaluation of our model on test dataset to check its performance.

        """
        self._filename = filename
        self.is_get_all_info = is_get_all_info
        self.offset_save_per_line = 10 # a trade-off between the loading speed and the memory usage
        self._total_size, self.offset_data = generate_json_for_csv_line_offset(filename, save_per_line=self.offset_save_per_line)
        self._is_csv_header = int(is_csv_header)
        with open(self._filename, 'r') as csv_file:
            line = csv_file.readline().strip()
            expected_header = 'query, ID, doc, click, purchase'
            if line == expected_header:
                if not is_csv_header:
                    logging.info('The first line of the file "%s" \t is \t "%s".' % (filename, line))
                    logging.info('It seems like this CSV file has header. Please set --is_csv_header')
                    raise ValueError
            else:
                if is_csv_header:
                    logging.info('The first line of the file "%s" \t is \t "%s".' % (filename, line))
                    logging.info('It seems like this CSV file does not have header. Please do not set --is_csv_header')
                    raise ValueError

        self._total_size -= self._is_csv_header # the CSV header is not a sample
        self.csv_reader_setting = {'delimiter':",", 'quotechar':'"', 'doublequote':False, 'escapechar':'\\', 'skipinitialspace':True}
        self.train_val_subblock_size = 10
        if val_ratio != 0: # train dataset
            val_size = int(self._total_size * val_ratio)
            np.random.seed(0)
            total_block_num = self._total_size // self.train_val_subblock_size
            val_block_num = val_size // self.train_val_subblock_size
            remaining_num = self._total_size % self.train_val_subblock_size
            self.val_indices = np.random.choice(range(total_block_num), \
                                                size=val_block_num, replace=False)
            val_indices_set = set(self.val_indices)
            self.idx_mapping = []
            for index_on_train_file in range(total_block_num + int(remaining_num != 0)):
                if index_on_train_file not in val_indices_set:
                    self.idx_mapping.append(index_on_train_file)
            self.idx_mapping = np.array(self.idx_mapping)
            self._total_size = (len(self.idx_mapping) - int(remaining_num != 0)) * self.train_val_subblock_size + remaining_num
            self.is_test_data = False
        elif val_indices is not None: # val dataset
            self.idx_mapping = val_indices
            self._total_size = len(self.idx_mapping) * self.train_val_subblock_size
            self.is_test_data = False
        else: # test dataset
            self.is_test_data = True

    def __getitem__(self, idx):
        """We fetch the sample (from train/val/test dataset) from the CSV file.
        If it is training or validation dataset, we use block-wise mapping to find the line number.
        
        Parameters
        ----------
        idx : int
            Which sample we are fetching.

        Returns
        -------
        parsed_list : list
            If is_get_all_info is True, return a list of string: [query, ID, document, click, purchase]
            If is_get_all_info is False (by default), return a list of strings: [query, document]
        """
        if not self.is_test_data:
            block_idx = idx // self.train_val_subblock_size
            inbloack_idx = idx % self.train_val_subblock_size
            block_idx = self.idx_mapping[block_idx]
            idx = block_idx * self.train_val_subblock_size + inbloack_idx
        idx += self._is_csv_header # the CSV header is not a sample
        offset_idx = idx // self.offset_save_per_line
        offset = self.offset_data[offset_idx]
        try:
            with open(self._filename, 'r') as csv_file:
                csv_file.seek(offset)
                for _ in range(1 + idx % self.offset_save_per_line):
                    line = csv_file.readline()
            line = line.replace('\0','')
            csv_line = csv.reader([line], **self.csv_reader_setting)
            parsed_list = next(csv_line) # [query, ID, document, click, purchase]
            if self.is_get_all_info:
                return parsed_list
            else:
                return [parsed_list[0], parsed_list[2]]
        except: # This is for quick inspection about which part goes wrong.
            error_message = "Something wrong when reading CSV samples.\n Details (filename, line index): {}, \t {}".format(self._filename, idx)
            raise IOError(error_message)
      
    def __len__(self):
        """The total number of samples in the dataset.
        """
        return self._total_size
