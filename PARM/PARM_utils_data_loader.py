# Utils functions for loading the data
import torch
import numpy as np
import random
import h5py
import os
import pandas as pd
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


def index_of_interest(
    data_length: int,
    file_path: str,
    stranded: bool,
    features: str,
    L_max: int,
    enhancers_slide_window=False,
    feat_selection_percentage=False,
    selection_length=False,
    order_length=False,
    L_min=False,
):
    """

    This function returns the index of the dataset that we are interested in.
    Based on stranded data and features of interest.

    Args:
        data_length: (int) data length of interest, the same as in file_path
        file_path: (str) Complete file path to h5 file of interest
        stranded: (bool) If we are interested in stranded fragments (True, only those matching with TSS)
                        or also complementary strands (False). Only applicable for TSS.
        features: (str) Features of interest that we want to select e.g. Enh, TSS, and/ or peaks.
                        If more than one feature is desired, please separate by "_" as in "TSS_peaks"

        feat_selection_percentage: (float) If not False, we select only a percentage of features given feat_selection_percentage*100.

        selection_length: (boolean) If True, it will downsample the fragment samples so that the length distirbutions is the same as the one
                                    in a numpy array provided in the same folder than the h5 file.


        order_length: (boolean) If True the index is returned in order of  fragment length

        L_max: (int) Used if there's a need to do a selection  (downsampling) based on fragment length


    Returns:
        index: (np.array) Index sample of interest.

    """

    dict_features_int = {
        "TSS": 0,
        "EnhA": 1,
        "peaks": 2,
        "EnhAmany": 3,
        "EnhAstrong": 4,
    }

    index = np.arange(data_length)

    index_features = np.array([])

    print(f"                     File path in index_of _interest function {file_path}")

    # Always selecct base on Features (TSS, EnhA, peaks, etc.)
    with h5py.File(file_path, "r") as file:
        feature_index = np.array(file["FEAT"]["FEATtype"])
        stranded_boolean_index = np.array(file["FEAT"]["FEATsense"]).astype(bool)

        for ind_feature in features.split("_"):  # loop thorugh the different features
            int_features = dict_features_int[ind_feature]

            if stranded and (ind_feature == "TSS"):
                index_features = np.append(
                    index_features,
                    index[(feature_index == int_features) & stranded_boolean_index],
                )

            else:
                index_features = np.append(
                    index_features, index[(feature_index == int_features)]
                )

    index = index_features.astype(int)

    index = np.sort(index)
    index = np.unique(index)

    # There was an error when generating the mouse data and some samples with INF or NAN values were saved, so we have to avoid taking these ones

    if "mouse" in file_path:
        with h5py.File(file_path, "r") as file:
            Y_index = np.array(file["Y"]["Log2Norm_mESC"][index])
            index = index[np.isfinite(Y_index)]

    # Get info of split
    split = os.path.basename(file_path).split(".hdf5")[0]
    directory_file = os.path.dirname(file_path)

    # Select fragments based on length, smaller than L_max and bigger than L_min
    if L_max or L_min:

        print(
            f"      Selecting fragments with length < {L_max} and >= {L_min}",
            flush=True,
        )

        size_file = os.path.join(directory_file, "..", "metadata", f"{split}_size.hdf5")

        if os.path.exists(size_file):

            with h5py.File(size_file, "r") as file:
                len_fragment = np.array(file["len"])[index]

                before_n_fragments = len(index)

                if L_min:
                    index = index[
                        (len_fragment < L_max) & (L_min <= len_fragment)
                    ]  # Check all indices that have this length
                else:
                    index = index[(len_fragment < L_max)]

            print(
                f"          Taking size file from {size_file} - Before n of fragments {before_n_fragments}, after {len(index)}",
                flush=True,
            )

            index = index.astype(int)

            index = np.sort(index)
            index = np.unique(index)

        else:
            raise Exception(
                f"   --------- File to check fragment length does not exists {size_file}",
                flush=True,
            )

    # Make distribution of length fragments similar to a predetermined
    if selection_length:

        distribution_desired = pd.read_csv(
            os.path.join(
                directory_file, "..", "distribution_length_fragments_genome_wide.csv"
            )
        ).sort_values("count", ascending=True)
        distribution_desired = distribution_desired[
            (distribution_desired["size"] > 126) & (distribution_desired["size"] < 569)
        ]

        step = 20
        distribution_desired = distribution_desired.groupby(
            pd.cut(
                distribution_desired["size"],
                np.arange(
                    distribution_desired["size"].min(),
                    distribution_desired["size"].max() + step,
                    step,
                ),
            )
        ).sum()
        distribution_desired["frac"] = (
            distribution_desired["size"] / distribution_desired["size"].sum()
        )
        distribution_desired = dict(
            zip(distribution_desired.index, distribution_desired["frac"])
        )

        total_samples = int(len(index) * 0.3)
        with h5py.File(
            os.path.join(directory_file, "..", "metadata", f"{split}_size.hdf5"), "r"
        ) as file:
            len_fragment = np.array(file["len"])[index]

            index_length = np.array([])
            for length in distribution_desired.keys():
                n_fragments = int(
                    distribution_desired[length] * total_samples
                )  # Number of fragments needed it for this length

                index_length_ind = index[
                    (len_fragment >= length.left) & (len_fragment < length.right)
                ]  # Check all indices that have this length

                if n_fragments > len(index_length_ind):
                    print(
                        f"Problem with length {length.left}-{length.right}: {n_fragments} Difference {(n_fragments-len(index_length_ind))/n_fragments*100}",
                        flush=True,
                    )
                index_length_ind = np.random.choice(index_length_ind, n_fragments)

                index_length = np.append(index_length, index_length_ind)

        index = index_length.astype(int)

        index = np.sort(index)
        index = np.unique(index)

    # Order index based on length
    if order_length:
        print("    Ordering training index by fragment length")

        with h5py.File(
            os.path.join(directory_file, "..", "metadata", f"{split}_size.hdf5"), "r"
        ) as file:
            len_fragment = np.array(file["len"], dtype="int")[index]

        # We could just order it but then it takes for ever to load the data during training
        index_length = np.array([])
        for len_split in range(0, len_fragment.max() + 50, 50):
            index_length_ind = np.unique(
                index[(len_fragment <= len_split) & (len_fragment > (len_split - 50))]
            )
            index_length = np.append(index_length, index_length_ind)

        index = index_length.astype(int)

    if feat_selection_percentage:
        # Which fragments to chose based on features (if they belong to TSS, EnhA, peaks, etc.)
        with h5py.File(os.path.join(file_path, "TSS.h5"), "r") as file:
            unique_TSS = np.unique(np.array(file["TSS"]))
            index_TSS = np.random.choice(
                unique_TSS,
                size=int(len(unique_TSS) * feat_selection_percentage),
                replace=False,
            )

            index_TSS = index[np.isin(np.array(file["TSS"]), index_TSS)]

    return index.astype(int)


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

        print_errors = False

        (xx, yy) = zip(*batch)

        # loop through the samples within the batch

        # Only change sequence or padding if any of these are true, otherwise just left
        ### the encoding as it is
        diff_padding = []

        for mini_batch in xx:
            for it, sample in enumerate(mini_batch):

                if print_errors:
                    print(f"sample  {sample.shape}\n", flush=True)
                # compute the length
                len_sample = int(sample.sum())
                if print_errors:
                    print(f"len_sample  {len_sample}\n", flush=True)

                start_len = int(np.ceil((sample.shape[0] - len_sample) / 2))
                end_len = start_len + len_sample

                if print_errors:
                    print(f"start_len  {start_len}   end_len {end_len}\n", flush=True)

                seq = sample[start_len:end_len, :]

                ##add adaptors if relevant
                if self.adaptor_3 is not False:
                    if (len_sample + self.adaptor_3.shape[0]) <= self.L_max:
                        len_sample += self.adaptor_3.shape[0]
                        seq = np.concatenate((seq, self.adaptor_3), 0)
                        if print_errors:
                            print(
                                f"After adding adaptor 3  {seq.shape}  \n", flush=True
                            )

                if self.adaptor_5 is not False:
                    if (len_sample + self.adaptor_5.shape[0]) <= self.L_max:
                        len_sample += self.adaptor_5.shape[0]
                        seq = np.concatenate((self.adaptor_5, seq), 0)
                        if print_errors:
                            print(
                                f"After adding adaptor 5  {seq.shape}  \n", flush=True
                            )

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


class ShuffleBatchSampler(torch.utils.data.sampler.BatchSampler):
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


class H5Dataset(torch.utils.data.Dataset):
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

    def __init__(self, path, celltype, Zscore_logTPM, barcode=False):
        self.file_path = path
        self.dataset = None
        self.celltype = celltype
        self.Zscore_logTPM = Zscore_logTPM
        self.barcode = barcode

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

                if self.barcode:
                    barcode = np.float32(
                        file["X"]["barcode"]["OneHotEncoding"][index_unique_file, :, :]
                    )
                    onehot_ind = np.stack([onehot_ind, barcode], axis=0)

                onehot.extend(list(onehot_ind))

                for it, cell in enumerate(multiple_cells):

                    replicates_cells = re.split(
                        r"(?<=T[1-9])_", cell
                    )  # If there are multiple replicates, split the ID but then average the score

                    if len(replicates_cells) > 1:
                        sure_it = []
                        for rep in replicates_cells:
                            sure_it.append(
                                np.float32(
                                    file["Y"][f"{self.Zscore_logTPM}_{rep}"][
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
                                file["Y"][f"{self.Zscore_logTPM}_{cell}"][
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
                        file["Y"][f"{self.Zscore_logTPM}_{replicate_cell}"]
                    )

        else:
            with h5py.File(os.path.join(self.file_path), "r") as file:
                dataset_len = len(file["Y"][f"{self.Zscore_logTPM}_{replicate_cell}"])

        return dataset_len


class GradualWarmupScheduler(_LRScheduler):
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
        super(GradualWarmupScheduler, self).__init__(optimizer)

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
            return super(GradualWarmupScheduler, self).step(epoch)
