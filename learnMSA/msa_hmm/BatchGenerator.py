from abc import ABC, abstractmethod

import numpy as np
import sys

from learnMSA.msa_hmm.SequenceDataset import SequenceDataset


class BatchGenerator(ABC):
    """
    A generator that generates batches of sequences from a dataset.
    """
    _data: SequenceDataset

    @property
    def data(self) -> SequenceDataset:
        return self._data

    @data.setter
    def data(self, data: SequenceDataset) -> None:
        self._data = data
        if self.is_shuffled():
            self.permutations = [
                np.arange(len(data)) 
                for _ in range(self.shuffle_batches)
            ]
            for p in self.permutations:
                np.random.shuffle(p)

    def __init__(
        self, 
        shuffle_batches : int = -1,
        crop_long_seqs : int = sys.maxsize,
        return_indices : bool = False, 
        return_crop_boundaries : bool = False,
        closing_terminal : bool = False,
        verbose : bool = False
    ):
        """
        Args:
            shuffle_batches (int): This many shuffled batches will be
                generated. Each shuffling maps indices to a different
                random permutation of the sequences in the dataset. If -1 
                (default), no shuffling is performed.
            crop_long_seqs (int): If > 0, sequences longer than this length
                will be cropped to this length (default: no cropping).
            return_indices (bool): If True, the selected indices of 
                the sequences in the dataset will be returned as well. This
                is useful when shuffling is enabled, as the indices do not
                map directly to the sequences in the dataset.
            return_crop_boundaries (bool): If True, the start and end indices 
                of the cropped sequences (i.e. sequences longer than 
                crop_long_seqs) will be returned as well.
            closing_terminal (bool): If True, all sequences in the batch will
                end with a terminal symbol.
            verbose (bool): If True, the batch generator will print additional
                information about its configuration and the generated batches.
        """
        self.shuffle_batches = shuffle_batches
        self.crop_long_seqs = crop_long_seqs
        self.return_indices = return_indices
        self.return_crop_boundaries = return_crop_boundaries
        self.closing_terminal = closing_terminal
        self.verbose = verbose

    def is_valid(self) -> bool:
        """ 
        Returns True if the batch generator is ready to be used.
        """
        return hasattr(self, '_data') 
    
    def is_shuffled(self) -> bool:
        """ 
        Returns True if the batch generator is configured to shuffle the 
        sequences.
        """
        return self.shuffle_batches >= 1
    
    def shuffle_dim(self) -> int:
        """ 
        Returns the dimension of the shuffle axis, i.e. the number of 
        shuffled batches.
        """
        return self.shuffle_batches if self.is_shuffled() else 1

    @abstractmethod
    def __call__(
        self, 
        indices : np.ndarray | list[int],
    ) -> np.ndarray | tuple[np.ndarray, ...]:
        """ 
        Generates a batch of sequences from the dataset based on the 
        provided indices.
            
        Args:
            indices (np.ndarray): Indices of the sequences to be included in 
                the batch. If shuffle is True, the indices to not map directly 
                to the sequences in the dataset, but to a permutated version 
                of the sequences. 

        Returns:
            A tuple containing the batch of sequences.
            The shape of the sequence batch is 
            (batch_size, length) if the dataset if not shuffled
            else it is (batch_size, shuffle_batches, length).
            Also returns the (permutated) indices and, if 
            return_crop_boundaries is True, the start and end indices of the 
            cropped sequences. If return_only_sequences is True, 
            only the batch of sequences is returned.
        """
        ...
    
    @abstractmethod
    def get_out_types(
            self
    ) -> tuple[type[np.generic], ...]:
        """ Returns the output types of the batch generator. """
        ...


class DefaultBatchGenerator(BatchGenerator):
    
    def __call__(
        self, 
        indices : np.ndarray | list[int], 
    ) -> np.ndarray | tuple[np.ndarray, ...]:
        
        if isinstance(indices, list):
            indices = np.array(indices, dtype=np.int64)

        if not self.is_valid():
            raise ValueError(
                "The batch generator is not ready to use." \
                " Please provide it with a sequence dataset first."
            ) 
        
        if self.is_shuffled():
            permutated_indices = np.stack(
                [perm[indices] for perm in self.permutations], axis=1
            )
        else:
            permutated_indices = indices[:, np.newaxis]

        max_len = np.max(self.data.seq_len(permutated_indices))
        max_len = min(max_len, self.crop_long_seqs)

        # initialize arrays
        batch_shape = (
            indices.shape[0], 
            self.shuffle_dim(), 
            max_len+int(self.closing_terminal)
        )
        batch = np.zeros(batch_shape, dtype=np.uint8) 
        #initialize the batch with terminal symbols
        batch += self.data.alphabet_size()-1
        if self.return_crop_boundaries:
            shape = (
                indices.shape[0], 
                self.shuffle_dim()
            )
            start = np.zeros(shape, dtype=np.int32)
            end = np.zeros(shape, dtype=np.int32)

        # fill the batch
        for i,perm_ind in enumerate(permutated_indices):
            for k,j in enumerate(perm_ind):
                if self.return_crop_boundaries:
                    seq, start[i, k], end[i, k] = self.data.get_encoded_seq(
                        j, 
                        crop_to_length=self.crop_long_seqs, 
                        return_crop_boundaries=True
                    )
                else:
                    seq = self.data.get_encoded_seq(
                        j, 
                        crop_to_length=self.crop_long_seqs, 
                        return_crop_boundaries=False
                    )
                m = min(self.data.seq_len(j), self.crop_long_seqs)
                batch[i, k, :m] = seq
        
        if not self.is_shuffled():
            # remove unused dimension
            batch = batch[:, 0] 
            permutated_indices = permutated_indices[:, 0]
        
        if self.return_indices:
            if self.return_crop_boundaries:
                return (batch, permutated_indices, start, end)
            else: 
                return (batch, permutated_indices)
        else:
            if self.return_crop_boundaries:
                return (batch, start, end)
            else: 
                return batch
            
    def get_out_types(self) -> tuple[type[np.generic], ...]:
        if self.return_indices:
            if self.return_crop_boundaries:
                return (np.uint8, np.int64, np.int32, np.int32)
            else:
                return (np.uint8, np.int64) 
        else:
            if self.return_crop_boundaries:
                return (np.uint8, np.int32, np.int32)
            else:
                return (np.uint8, )