import numpy as np
import copy
import math
from Bio import SeqIO, SeqRecord
import re
from functools import partial


class SequenceDataset:
    """ Contains a set of sequences and their corresponding labels.
    """
    #alphabet[:20] corresponds to the traditional aminoacid alphabet
    #for future changes in the alphabet: all learnMSA related code assumes that the standard amino acids occur at the first 20 positions
    #any special symbols should be added after that and the gap character comes last
    alphabet = "ARNDCQEGHILKMFPSTWYVXUO-"
    


    def __init__(self, filename=None, fmt="fasta", sequences=None, indexed=False, threads=None):
        """
        Args:
            filename: Path to a sequence file in any supported format.
            fmt: Format of the file. Can be any format supported by Biopython's SeqIO.
            sequences: A list of id/sequence pairs as strings. If given, filename and fmt arguments are ignored.
            indexed: If True, Biopython's index method is used to avoid loading the whole file into memory at once. Otherwise 
                    regular parsing is used. Setting this to True will allow constant memory training at the cost of per-step performance.
            threads: Number of threads to use for metadata computation.
        """
        if sequences is None and filename is None:
            raise ValueError("Either filename or sequences must be given.")
        if sequences is None:
            self.filename = filename
            self.fmt = fmt
            self.indexed = indexed
            try:
                if indexed:
                    self.record_dict = SeqIO.index(filename, fmt)
                else:
                    self.record_dict = SeqIO.to_dict(SeqIO.parse(filename, fmt))
                self.parsing_ok = True
            except ValueError as err:
                self.parsing_ok = False
                # hold the error and raise it when calling validate_dataset
                self.err = err
            if not self.parsing_ok:
                return
        else:
            self.parsing_ok = True
            self.filename = ""
            self.fmt = ""
            self.indexed = False
            self.record_dict = {s[0] : SeqRecord.SeqRecord(s[1], id=s[0]) for s in sequences}
        self.seq_ids = list(self.record_dict)
        self.num_seq = len(self.seq_ids)
        self.seq_lens = np.array([sum([1 for x in str(self.get_record(i).seq) if x.isalpha()]) for i in range(self.num_seq)])
        self.max_len = np.amax(self.seq_lens) if self.seq_lens.size > 0 else 0


    def __enter__(self):
        return self


    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


    def close(self):
        if self.indexed:
            self.record_dict.close()


    def get_record(self, i):
        return self.record_dict[self.seq_ids[i]]


    def get_alphabet_no_gap(self):
        return type(self).alphabet[:-1]


    def get_standardized_seq(self, i,
                            remove_gaps=True, 
                            gap_symbols="-.", 
                            ignore_symbols="", 
                            replace_with_x = ""): 
        seq_str = str(self.get_record(i).upper().seq)
        # replace non-standard aminoacids with X
        for aa in replace_with_x:
            seq_str = seq_str.replace(aa, 'X')
        if remove_gaps:
            for s in gap_symbols:
                seq_str = seq_str.replace(s, '')
        else:
            # unify gap symbols
            for s in gap_symbols:
                seq_str = seq_str.replace(s, gap_symbols[0])
        # strip other symbols
        for s in ignore_symbols:
            seq_str = seq_str.replace(s, '')
        return seq_str



    def get_encoded_seq(self, i, 
                        remove_gaps=True, 
                        gap_symbols="-.", 
                        ignore_symbols="", 
                        replace_with_x = "BZJ", 
                        crop_to_length=math.inf,
                        validate_alphabet=True, 
                        dtype=np.int16,
                        return_crop_boundaries=False):
        seq_str = self.get_standardized_seq(i, remove_gaps, gap_symbols, ignore_symbols, replace_with_x)
        # make sure the sequences do not contain any other symbols
        if validate_alphabet:
            if bool(re.compile(rf"[^{type(self).alphabet}]").search(seq_str)):
                raise ValueError(f"Found unknown character(s) in sequence {self.seq_ids[i]}. Allowed alphabet: {type(self).alphabet}.")
        seq = np.array([type(self).alphabet.index(aa) for aa in seq_str], dtype=dtype)
        if seq.shape[0] > crop_to_length:
            #crop randomly
            start = np.random.randint(0, seq.shape[0] - crop_to_length + 1)
            end = start + crop_to_length
        else:
            start = 0
            end = seq.shape[0]
        seq = seq[start:end]
        if return_crop_boundaries:
            return seq, start, end
        else:
            return seq
     
        
    def validate_dataset(self, single_seq_ok=False, empty_seq_id_ok=False, dublicate_seq_id_ok=False):
        if not self.parsing_ok:
            raise self.err

        """ Raise an error if something unexpected is found in the sequences. """
        if len(self.seq_ids) == 1 and not single_seq_ok:
            raise ValueError(f"File {self.filename} contains only a single sequence.") 
            
        if len(self.seq_ids) == 0:
            raise ValueError(f"Could not parse any sequences from {self.filename}.") 

        if np.amin(self.seq_lens) == 0:
            raise ValueError(f"{self.filename} contains empty sequences.") 

        if not empty_seq_id_ok:
            for sid in self.seq_ids:
                if sid == '':
                    raise ValueError(f"File {self.filename} contains an empty sequence ID, which is not allowed.") 
        if len(self.seq_ids) > len(set(self.seq_ids)) and not dublicate_seq_id_ok:
            raise ValueError(f"File {self.filename} contains duplicated sequence IDs. learnMSA requires unique sequence IDs.") 



class AlignedDataset(SequenceDataset):
    """ A sequence dataset with MSA metadata.
    Args:
        See SequenceDataset.
    """
    def __init__(self, filename=None, fmt="fasta", aligned_sequences=None, indexed=False, threads=None, single_seq_ok=False):
        super().__init__(filename, fmt, aligned_sequences, indexed, threads)
        self.single_seq_ok = single_seq_ok
        self.validate_dataset()
        self.msa_matrix = np.zeros((self.num_seq, len(self.get_record(0))), dtype=np.int16)
        for i in range(self.num_seq):
            self.msa_matrix[i,:] = self.get_encoded_seq(i, remove_gaps=False, dtype=np.int16)
        # compute a mapping from sequence positions to MSA-column index
        cumsum = np.cumsum(self.msa_matrix != type(self).alphabet.index('-'), axis=1)  #A-B--C -> 112223
        diff = np.diff(np.insert(cumsum, 0, 0.0, axis=1), axis=1) #112223 -> 0112223 -> [[(i+1) - i]] -> 101001
        diff_where = [np.argwhere(diff[i,:]).flatten() for i in range(diff.shape[0])]
        self.column_map = np.concatenate(diff_where).flatten()
        self.starting_pos = np.cumsum(self.seq_lens)
        self.starting_pos[1:] = self.starting_pos[:-1]
        self.starting_pos[0] = 0
        self.alignment_len = self.msa_matrix.shape[1]


    def validate_dataset(self):
        super().validate_dataset(single_seq_ok=self.single_seq_ok, empty_seq_id_ok=False, dublicate_seq_id_ok=False)
        record_lens = np.array([len(self.get_record(i)) for i in range(self.num_seq)])
        if np.any(record_lens != record_lens[0]):
            raise ValueError(f"File {self.filename} contains sequences of different lengths.")


    def get_column_map(self, i):
        s = self.starting_pos[i]
        e = s + self.seq_lens[i]
        return self.column_map[s:e]


    def SP_score(self, ref_data : "AlignedDataset", batch=512):
        total_len = sum(self.seq_lens)
        n = 0
        true_positives = 0
        self_positives = 0
        ref_positives = 0
        while n < self.column_map.shape[0]:
            self_homologs = np.expand_dims(self.column_map,0)==np.expand_dims(self.column_map[n:n+batch],1)
            ref_homologs = np.expand_dims(ref_data.column_map,0)==np.expand_dims(ref_data.column_map[n:n+batch],1)
            true_positives += np.sum(np.logical_and(self_homologs, ref_homologs)) 
            self_positives += np.sum(self_homologs)
            ref_positives += np.sum(ref_homologs)
            n+=batch
        true_positives -= total_len
        sp = true_positives / max(1, ref_positives - total_len)
        return sp