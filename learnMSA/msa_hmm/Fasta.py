import numpy as np
import copy



#alphabet[:20] corresponds to the traditional aminoacid alphabet
alphabet = ['A', 'R',  'N',  'D',  'C',  'Q',  'E',  'G',  'H', 'I',  'L',  'K',  
            'M',  'F',  'P', 'S',  'T',  'W',  'Y',  'V',  'B',  'Z',  'X', 'U', 'O', '$']
s = len(alphabet)
    
ind_dtype = np.uint16

def replace_in_strings(seqs, c_old, c_new):
    for i in range(len(seqs)):
        seqs[i] = seqs[i].replace(c_old, c_new)
    
    
# a class that reads sequences in fasta file format 
# condensed representation of a potentially large sequence datatset
# all sequences are stored in a flat array of 16bit integers
# one-hot representations are only constructed on the fly for small batches
class Fasta:
    def __init__(self, 
                 filename, #fasta file to parse
                 aligned = False #automatically assumes an alignment if gaps are found, this flag only has to be set manually, if the file contains a gapless alignment
                ):
        self.filename = filename
        self.aligned = aligned
        self.read_seqs(filename)
        if self.gaps:
            self.compute_targets()

            
            
    def read_seqs(self, filename):
        #read seqs as strings
        with open(filename) as f:
            content = f.readlines()
        self.seq_ids = []
        self.raw_seq = []
        for line in content:
            line = line.strip()
            if len(line)>0:
                if line[0]=='>':
                    self.seq_ids.append(line[1:])
                    self.raw_seq.append("")
                elif len(self.raw_seq) > 0:
                    self.raw_seq[-1] += line
                        
                    
        self.gaps = self.validate()
        
        if self.gaps:    
            self.alignment_len = len(self.raw_seq[0])
            replace_in_strings(self.raw_seq, '.', '-')
            
        for i,c in enumerate(alphabet[:-1]):
            replace_in_strings(self.raw_seq, c, str(i)+' ')
            replace_in_strings(self.raw_seq, c.lower(), str(i)+' ')

        #can store sequences with gaps as matrix
        if self.gaps:
            self.ref_seq = copy.deepcopy(self.raw_seq)
            self.ref_seq = [s.replace('-',str(len(alphabet)-1)+' ') for s in self.ref_seq]
            self.ref_seq = np.reshape(np.fromstring("".join(self.ref_seq), dtype=int, sep=' '), (len(self.ref_seq), self.alignment_len))
            replace_in_strings(self.raw_seq, '-', '')
            
        self.raw_seq = [np.fromstring(s, dtype=ind_dtype, sep=' ') for s in self.raw_seq]
        #concatenate to avoid memory fragmentation
        self.seq_lens = np.array([s.shape[0] for s in self.raw_seq])
        self.raw_seq = np.concatenate(self.raw_seq, axis=0)
        self.starting_pos = np.cumsum(self.seq_lens)
        self.starting_pos[1:] = self.starting_pos[:-1]
        self.starting_pos[0] = 0
        self.total_len = np.sum(self.seq_lens)
        self.max_len = np.amax(self.seq_lens)
        self.num_seq = len(self.seq_lens)
        #also store the permutation of sequence indices that sorts by length
        self.sorted_indices = np.array([i for l,i in sorted(zip(self.seq_lens, range(self.num_seq)))]) 
        
        
    def validate(self):
        gaps = self.aligned
        if len(self.raw_seq) == 1:
            raise SystemExit(f"File {self.filename} contains only a single sequence.") 
            
        if len(self.raw_seq) == 0:
            raise SystemExit(f"Can not read sequences from file {self.filename}. Expected a file in FASTA format containing at least 2 sequences.") 
                    
        #validate seq ids (required for anc.probs. to work correctly)
        if len(self.raw_seq) != len(self.seq_ids):
            raise SystemExit(f"Can not parse file {self.filename}. Please check if the FASTA format is correct.")
        for sid in self.seq_ids:
            if sid == "":
                raise SystemExit(f"File {self.filename} contains an empty sequence ID, which is not allowed.") 
        if len(self.seq_ids) > len(set(self.seq_ids)):
            raise SystemExit(f"File {self.filename} contains duplicated sequence IDs. learnMSA requires unique sequence IDs.") 
            
        
        #check for alphabet problems and empty sequences
        for sid, seq in zip(self.seq_ids, self.raw_seq):
            if len(seq) == 0:
                raise SystemExit(f"File {self.filename} contains an empty sequence: \'{sid}\'.") 
            for aa in alphabet[:-1]:
                seq = seq.replace(aa, "")
                seq = seq.replace(aa.lower(), "")
            if not seq == "":
                unique = "".join(set(seq))
                if "-" in unique or "." in unique:
                    gaps = True
                for gap in ["-", "."]:
                    seq = seq.replace(gap, "")
                if not seq == "":
                    alphabet_str = "".join(alphabet[:-1])
                    raise SystemExit(f"In file {self.filename}: Found unknown character(s) {unique} in sequence \'{sid}\'. Allowed alphabet: {alphabet_str}.") 
               
        if gaps:
            #validate alignment
            for seq in self.raw_seq[1:]:
                if len(seq) != len(self.raw_seq[0]):
                    raise SystemExit(f"In file {self.filename}: Although they contain gaps, the input sequences have different lengths. The file seems to contain a malformed alignment.") 
        return gaps
    
    
    def get_raw_seq(self, i):
        s = self.starting_pos[i]
        e = s + self.seq_lens[i]
        return self.raw_seq[s:e]

    
    #converts (a subset of) the sequences to one hot encodings
#     def one_hot_sequences(self, subset=None):
#         if subset is None:
#             subset = list(range(len(self.seq_lens)))
#         num_seqs = len(subset)
#         lens = [self.seq_lens[si] for si in subset]
#         maxlen = max(lens)
#         alphabet_size = len(alphabet)-1
#         seq = np.zeros((num_seqs, maxlen, alphabet_size), dtype=np.float32)
#         for j,(l,si) in enumerate(zip(lens, subset)):
#             lrange = np.arange(l)
#             seq[j, lrange, self.get_raw_seq(si)] = 1
#         return seq
    


    def compute_targets(self):
        
        #a mapping from raw position to column index
        #A-B--C -> 112223
        cumsum = np.cumsum(self.ref_seq != len(alphabet)-1, axis=1) 
        #112223 -> 0112223 -> [[(i+1) - i]] -> 101001
        diff = np.diff(np.insert(cumsum, 0, 0.0, axis=1), axis=1) 
        diff_where = [np.argwhere(diff[i,:]).flatten() for i in range(diff.shape[0])]
        self.membership_targets = np.concatenate(diff_where).flatten()
      
    
    
    #returns a binary matrix indicating which residues correspond to which columns
    #if only a subset of the sequences is required, empty columns will be removed
    #output shape is (num_seq, num_columns, max_len_seq)
#     def column_memberships(self, subset=None):
#         if subset is None:
#             subset = list(range(len(self.raw_seq)))
        
#         lens = [self.seq_lens[si] for si in subset]
#         maxlen = max(lens)
#         num_seqs = len(subset)
        
#         #remove empty columns 
#         col_sizes = np.zeros(self.alignment_len)
#         for j,(l, si) in enumerate(zip(lens, subset)):
#             suml = sum(self.seq_lens[:si])
#             col_sizes[self.membership_targets[suml:(suml+l)]] += 1
#         empty = (col_sizes == 0)
#         num_columns = int(np.sum(~empty))
#         cum_cols = np.cumsum(empty)
            
#         #shift column indices according to removed empty columns
#         corrected_targets = []
#         target_subset = np.zeros(self.alignment_len, dtype=bool)
#         for j,(l, si) in enumerate(zip(lens, subset)):
#             suml = sum(self.seq_lens[:si])
#             ct = self.membership_targets[suml:(suml+l)]
#             target_subset[ct] = 1
#             corrected_targets.append(ct - cum_cols[ct])

#         memberships = np.zeros((num_seqs, maxlen, num_columns), dtype=np.float32)
#         for j,(l, targets) in enumerate(zip(lens, corrected_targets)):
#             lrange = np.arange(l)
#             memberships[j, lrange, targets] = 1
            
            
#         return np.transpose(memberships, [0,2,1]), target_subset
            
        
    
    def aminoacid_seq_str(self, i):
        seq = ""
        for j in self.get_raw_seq(i):
            seq += alphabet[j]
        return seq
    
    
    def column_str(self, i):
        col = ""
        for j in self.ref_seq[:,i]:
            col += alphabet[j]
        return col
    
    
    #equivalent to modeler and developer/SP score respectively
    #batch size can be reduced to resolve memory issues, it does not affect the result
    def precision_recall(self, ref_fasta, batch=512, verbose=False):
        total_len = sum(self.seq_lens)
        n = 0
        true_positives = 0
        self_positives = 0
        ref_positives = 0
        while n < self.membership_targets.shape[0]:
            if verbose:
                print(n)
            self_homologs = np.expand_dims(self.membership_targets,0)==np.expand_dims(self.membership_targets[n:n+batch],1)
            ref_homologs = np.expand_dims(ref_fasta.membership_targets,0)==np.expand_dims(ref_fasta.membership_targets[n:n+batch],1)
            true_positives += np.sum(np.logical_and(self_homologs, ref_homologs)) 
            self_positives += np.sum(self_homologs)
            ref_positives += np.sum(ref_homologs)
            n+=batch
        true_positives -= total_len
        prec = true_positives / max(1, self_positives - total_len)
        recall = true_positives / max(1, ref_positives - total_len)
        return prec, recall
    
    
    def positions_in_columns(self):
        pos = -np.ones((len(self.seq_lens), self.alignment_len), dtype=np.int32)
        l_sum = 0
        for i,l in enumerate(self.seq_lens):
            pos[i, self.membership_targets[l_sum:l_sum+l]] = np.arange(l, dtype=np.int32)
            l_sum += l
        return pos
    
    
    #no. common columns between test (self) and reference divided by the no. cols. in the ref.
    #a common column is defined by positions not by mere amino acid equality
    #batch size can be reduced to resolve memory issues, it does not affect the result
    def tc_score(self, ref_fasta, batch=32):
        test_pos = self.positions_in_columns()
        ref_pos = ref_fasta.positions_in_columns()
        #simple quadratic time test
        correct_columns = 0
        n = 0
        while n < ref_fasta.alignment_len:
            pairwise = np.expand_dims(test_pos, -1) == np.expand_dims(ref_pos[:, n:n+batch], -2)
            correct_columns += np.sum(np.all(pairwise, 0))
            n += batch
        return correct_columns / ref_fasta.alignment_len
    
