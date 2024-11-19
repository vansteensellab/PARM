# Utils functions for loading the data
import torch
import numpy as np
import random
import h5py
import os
import re
from torch.optim.lr_scheduler import _LRScheduler


def getOneHot(Seqs, L_max=False, padding_value=0):
    """
    One hot encode in a torch tensor a sequence of DNA. Pad with 0 if necessary til L_max is reached.
    Args:
         Seqs: (list) of all sequences to one hot encode. If only one seq, use list as well.
         L_max: (int) Maximum length of sequence that Model can take.
         padding_value: (int or float) Value to pad with. (default: 0)

    Returns:
         X_OneHot: (torch.tensor) sequence one hot encoded with shape (n_sequences, nt, length_sequnce)
    """

    # Define nucleotide to vector
    letter2vector = {
        "A": np.array([1.0, 0.0, 0.0, 0.0]),
        "C": np.array([0.0, 1.0, 0.0, 0.0]),
        "G": np.array([0.0, 0.0, 1.0, 0.0]),
        "T": np.array([0.0, 0.0, 0.0, 1.0]),
        "N": np.array([0.0, 0.0, 0.0, 0.0]),
    }

    if not L_max and type(Seqs) != list:
        x = np.array([letter2vector[s] for s in Seqs])

        x = torch.from_numpy(x)

        return x

    # get On Hot Encoding
    X_OneHot = []
    for seq in Seqs:
        seq = seq.upper()

        x = np.array([letter2vector[s] for s in seq])
        pw = (L_max - x.shape[0]) / 2
        PW = [int(np.ceil(pw)), int(np.floor(pw))]
        X_OneHot.append(np.pad(x, [PW, [0, 0]], constant_values=padding_value))
    X_OneHot = np.array(X_OneHot)

    return X_OneHot


class pad_collate:
    """
    Transforms the data padded in the center in to padding in the left and
        padding in the right. It returns, mid, left, and right padding randomly from samples in batch.
    Args:
        batch: (torch.tensor) Samples torch matrix with shape (batch, voc_length, seq_len)
        L_max: (int) max length of the sequence to be padded
        adaptor_5: (str) If not False, sequence to pad in the left, right folowed by the sequence.
        adaptor_3: (str) If not False, sequence to pad in the right, right after to the sequence.

    """

    def __init__(
        self, L_max=600, adaptor_5=False, adaptor_3=False, alternative_padding=True
    ):
        self.L_max = L_max

        self.adaptor_3 = adaptor_3
        self.adaptor_5 = adaptor_5
        self.L_max = L_max
        self.alternative_padding = alternative_padding

        if self.adaptor_3 is not False:
            self.adaptor_3 = getOneHot(adaptor_3)

        if self.adaptor_5 is not False:
            self.adaptor_5 = getOneHot(adaptor_5)

    def __call__(self, batch):
        """
        Transforms the data padded in the center in to padding in the left and
            padding in the right. It returns, mid, left, and right padding randomly from samples in batch.
        Args:
            batch: (torch.tensor) Samples torch matrix with shape (batch, voc_length, seq_len)
            L_max: (int) max length of the sequence to be padded
            adaptor_5: (str) If not False, sequence to pad in the left, right folowed by the sequence.
            adaptor_3: (str) If not False, sequence to pad in the right, right after to the sequence.

        """

        # Adaptor_3 Acgact
        # adaptor_5 agtgat

        # Input: list of tuples

        (xx, yy) = zip(*batch)

        # loop through the samples within the batch

        # Only change sequence or padding if any of these are true, otherwise just left
        ### the encoding as it is
        diff_padding = []

        for mini_batch in xx:
            for _, sample in enumerate(mini_batch):

                # compute the length
                len_sample = int(sample.sum())
                
                start_len = int(np.ceil((sample.shape[0] - len_sample) / 2))
                end_len = start_len + len_sample
                
                seq = sample[start_len:end_len, :]

                ##add adaptors if relevant
                if self.adaptor_3 is not False:
                    if (len_sample + self.adaptor_3.shape[0]) <= self.L_max:
                        len_sample += self.adaptor_3.shape[0]
                        seq = np.concatenate((seq, self.adaptor_3), 0)
                        
                if self.adaptor_5 is not False:
                    if (len_sample + self.adaptor_5.shape[0]) <= self.L_max:
                        len_sample += self.adaptor_5.shape[0]
                        seq = np.concatenate((self.adaptor_5, seq), 0)
                        
                if self.alternative_padding:

                    # Select a random start where sequence should be placed within an empty matrix
                    total_padding = self.L_max - len_sample

                    padding_left = random.randint(0, total_padding)
                    padding_right = self.L_max - len_sample - padding_left

                    PW = [padding_left, padding_right]

                else:  # If not alternative padding just put the sequence in the center
                    pw = (self.L_max - len_sample) / 2
                    PW = [int(np.ceil(pw)), int(np.floor(pw))]

                diff_padding.append(np.pad(seq, [PW, [0, 0]]))

        xx = torch.Tensor(np.stack(diff_padding, axis=0))
        yy = torch.Tensor(np.concatenate(list(yy), axis=0)).unsqueeze(1)

        return [xx, yy]


class shuffle_batch_sampler(torch.utils.data.sampler.BatchSampler):
    """Wraps another sampler to yield a mini-batch of indices.
    The `ShuffleBatchSampler` adds `shuffle` on top of `torch.utils.data.sampler.BatchSampler`.
    Args:
        sampler (Sampler): Base sampler.
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if its size would be
            less than ``batch_size``.
        shuffle (bool, optional): If ``True``, the sampler will shuffle the batches.
    """

    def __init__(self, sampler, batch_size, drop_last, shuffle=True):
        self.shuffle = shuffle
        super().__init__(sampler, batch_size, drop_last)

    def __iter__(self):
        # NOTE: This is not data
        batches = list(super().__iter__())
        if self.shuffle:
            random.shuffle(batches)
        return iter(batches)


class h5_dataset(torch.utils.data.Dataset):
    """Dataset of sequential data to train memory.
    Args:
        dataset_path (str or list): Path to HDF5 dataset file. List if multiple files
        celltype (str): Possible cell type (K562, HEPG2, hNPC) if more than one separate by "_".
                           If multiple, the order of this string will determine the order of the output labels.
        Zscore_logTPM (str): Either "Zscore" or "logTPM".

    Note:
        Arrays should have the same size of the first dimension and their type should be the
        same as desired Tensor type.
    """

    def __init__(self, path, celltype, measurement_column="Log2RPM_K562"):
        self.file_path = path
        self.dataset = None
        self.celltype = celltype
        self.measurement_column = measurement_column
        
        # Check if multiple cell types
        self.multiple_cells = self.celltype.split("__")

    def __getitem__(self, index):

        multiple_cells = self.multiple_cells

        # Index comes in a batch as batchsampler is used

        index = np.array(index)

        sure, onehot = [], []

        for unique_file_index in np.unique(
            index[:, 1]
        ):  # If there are different files in the same batch take the correct file for each sample

            file_sure = []

            # unique_file_index - Indicates the index of the files

            file_path = self.file_path[
                unique_file_index
            ]  # Get the path of the file given the index
            index_unique_file = np.sort(
                index[(index[:, 1] == unique_file_index), 0]
            )  # Order file so it opens first one and then the other and not intercalate

            with h5py.File(file_path, "r") as file:
                onehot_ind = np.float32(
                    file["X"]["sequence"]["OneHotEncoding"][index_unique_file, :, :]
                )

                
                onehot.extend(list(onehot_ind))

                for _, cell in enumerate(multiple_cells):

                    replicates_cells = re.split(
                        r"(?<=T[1-9])_", cell
                    )  # If there are multiple replicates, split the ID but then average the score

                    if len(replicates_cells) > 1:
                        sure_it = []
                        for rep in replicates_cells:
                            sure_it.append(
                                np.float32(
                                    file["Y"][f"{self.measurement_column}"][
                                        index_unique_file
                                    ]
                                )
                            )
                        sure_it = np.mean(np.array(sure_it), axis=0)

                    else:

                        if any(
                            cell in s for s in file["Y"].keys()
                        ):  # check if inded this dataset has the cellline we are looking for

                            sure_it = np.float32(
                                file["Y"][f"{self.measurement_column}"][
                                    index_unique_file
                                ]
                            )

                        else:
                            raise ValueError(
                                f"From multiple cell lines, the following one was not present in the dataset {cell}"
                                )

                    file_sure.append(sure_it)

            sure.append(np.array(file_sure))

        sure = np.concatenate(sure, axis=1)
        sure = np.transpose(sure)

        return onehot, sure

    def __len__(self):
        multiple_cells = self.multiple_cells[0]

        replicate_cell = re.split(r"(?<=T[1-9])_", multiple_cells)[
            0
        ]  # If there are multiple replicates, split the ID but then average the score

        if type(self.file_path) == list:  # If true more than one file
            dataset_len = 0

            for file in self.file_path:
                with h5py.File(file, "r") as file:
                    dataset_len += len(
                        file["Y"][f"{self.measurement_column}"]
                    )

        else:
            with h5py.File(os.path.join(self.file_path), "r") as file:
                dataset_len = len(file["Y"][f"{self.measurement_column}"])

        return dataset_len


class gradual_warmup_scheduler(_LRScheduler):
    """Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier if multiplier > 1.0. if multiplier = 1.0, lr starts from 0 and ends up with the base_lr.
        total_epoch: target learning rate is reached at total_epoch, gradually
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    """

    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        self.multiplier = multiplier
        if self.multiplier < 1.0:
            raise ValueError("multiplier should be greater thant or equal to 1.")
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super(gradual_warmup_scheduler, self).__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [
                        base_lr * self.multiplier for base_lr in self.base_lrs
                    ]
                    self.finished = True
                return self.after_scheduler.get_last_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        if self.multiplier == 1.0:
            return [
                base_lr * (float(self.last_epoch) / self.total_epoch)
                for base_lr in self.base_lrs
            ]
        else:
            return [
                base_lr
                * ((self.multiplier - 1.0) * self.last_epoch / self.total_epoch + 1.0)
                for base_lr in self.base_lrs
            ]

    def step(self, epoch=None, metrics=None):
        if self.finished and self.after_scheduler:
            if epoch is None:
                self.after_scheduler.step(None)
            else:
                self.after_scheduler.step(epoch - self.total_epoch)
            self._last_lr = self.after_scheduler.get_last_lr()
        else:
            return super(gradual_warmup_scheduler, self).step(epoch)
