# Test Conversion Guide: unittest to pytest

This guide documents the conversion pattern from unittest to pytest for all test classes in `test/UnitTests.py`.

## Conversion completed:
1. ✅ `TestDataset` → `tests/test_dataset.py`
2. ✅ `TestMsaHmmCell` → `tests/test_msa_hmm_cell.py`

## Remaining test classes to convert:

### 3. TestMSAHMM (lines 461-1028)
- **File:** `tests/test_msa_hmm.py`
- **Methods:** test_matrices, test_cell, test_viterbi, test_parallel_viterbi, test_aligned_insertions, test_backward, test_posterior_state_probabilities (duplicate), test_sequence_weights
- **Special notes:** 
  - Contains helper function `string_to_one_hot` and `get_all_seqs`
  - Has `assert_vec` method that should become a helper function
  - Very large test with extensive reference data

### 4. TestAncProbs (lines 1029-1246)
- **File:** `tests/test_anc_probs.py`
- **Methods:** test_paml_parsing, test_rate_matrices, test_anc_probs, test_encoder_model, test_transposed
- **Special notes:**
  - Has `__init__` with `self.paml_all` and `self.A` → convert to pytest fixture
  - Multiple helper methods: `assert_vec`, `parse_a`, `assert_equilibrium`, `assert_symmetric`, `assert_rate_matrix`, `assert_anc_probs`, `assert_anc_probs_layer`, `get_test_configs`, `get_simple_seq`

### 5. TestData (lines 1247-1268)
- **File:** `tests/test_data.py`
- **Methods:** test_default_batch_gen
- **Special notes:** 
  - Has `assert_vec` helper method

### 6. TestModelSurgery (lines 1269-1571)
- **File:** `tests/test_model_surgery.py`
- **Methods:** test_insert_col, test_remove_col, test_concat_models, test_change_length, test_change_num_models, test_slicing, test_remove_gaps, test_mask_func, test_mask_alignment
- **Special notes:**
  - Has `assert_vec` and `make_test_alignment` helper methods
  - Uses `AlignedDataset`

### 7. TestAlignment (lines 1572-1767)
- **File:** `tests/test_alignment.py`
- **Methods:** test_insert_gaps, test_decode_msa, test_decode_msa_lm, test_pairwise_alignment, test_guide_tree_fasta, test_guide_tree_stockholm, test_progressive_alignment
- **Special notes:**
  - Has `assert_vec` helper method
  - Extensive use of file I/O and MSA operations

### 8. TestPriors (lines 1768-1784)
- **File:** `tests/test_priors.py`
- **Methods:** test_dirichlet_mixture
- **Special notes:** Simple test class

### 9. TestModelToFile (lines 1785-1897)
- **File:** `tests/test_model_to_file.py`
- **Methods:** test_model_to_file, test_alignment_model, test_reconstruct_model, test_load_weights_to_new_model, test_emission_only
- **Special notes:**
  - File I/O operations
  - Model serialization/deserialization

### 10. TestMvnMixture (lines 1898-1975)
- **File:** `tests/test_mvn_mixture.py`
- **Methods:** test_mvn_mixture, test_mvn_mixture_from_data, test_bayes_theorem, test_em
- **Special notes:**
  - Statistical tests
  - EM algorithm testing

### 11. TestLanguageModelExtension (lines 1976-2030)
- **File:** `tests/test_language_model_extension.py`
- **Methods:** test_full_pipeline, test_cache, test_cache_overwrite
- **Special notes:**
  - Language model integration
  - Caching functionality

### 12. TestEmbeddingPretrainingDatapipeline (lines 2031-2038)
- **File:** `tests/test_embedding_pretraining_datapipeline.py`
- **Methods:** test_datapipeline
- **Special notes:** Simple test class

### 13. TestPretrainingUtilities (lines 2039-end)
- **File:** `tests/test_pretraining_utilities.py`
- **Methods:** test_pretraining, test_mask_function
- **Special notes:** Pretraining functionality

## Conversion Pattern

### 1. Replace class-based structure with module-level functions:
```python
# BEFORE (unittest)
class TestExample(unittest.TestCase):
    def test_something(self):
        self.assertEqual(a, b)

# AFTER (pytest)
def test_something():
    assert a == b
```

### 2. Convert `__init__` to pytest fixtures:
```python
# BEFORE
class TestExample(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data = load_data()
    
    def test_use_data(self):
        self.assertEqual(len(self.data), 10)

# AFTER
@pytest.fixture
def test_data():
    return load_data()

def test_use_data(test_data):
    assert len(test_data) == 10
```

### 3. Convert assertions:
- `self.assertEqual(a, b)` → `assert a == b`
- `self.assertTrue(x)` → `assert x`
- `self.assertFalse(x)` → `assert not x`
- `self.assertIsNone(x)` → `assert x is None`
- `self.assertRaises(Exception)` → `with pytest.raises(Exception):`
- Keep `np.testing.assert_*` as is

### 4. Convert helper methods to module-level functions:
```python
# BEFORE
class TestExample(unittest.TestCase):
    def assert_vec(self, x, y):
        self.assertEqual(x.shape, y.shape)

# AFTER
def assert_vec(x, y):
    assert x.shape == y.shape
```

### 5. Add imports:
```python
import os
import warnings  # if needed for warning suppression
import numpy as np
import pytest
import tensorflow as tf

from learnMSA.msa_hmm import ...  # relevant imports
```

## Example: Complete conversion of a small test class

```python
# BEFORE (unittest)
class TestData(unittest.TestCase):
    def assert_vec(self, x, y):
        self.assertEqual(x.shape, y.shape)
        self.assertTrue(np.all(x == y))

    def test_default_batch_gen(self):
        data = load_data()
        self.assert_vec(data, expected)

# AFTER (pytest)
def assert_vec(x, y):
    assert x.shape == y.shape
    assert np.all(x == y)

def test_default_batch_gen():
    data = load_data()
    assert_vec(data, expected)
```

## Notes for specific test classes:

1. **TestMSAHMM**: Large class with many helper functions - make sure to convert all helpers to module-level
2. **TestAncProbs**: Complex initialization - use class-based fixture or module-level constants
3. **TestModelSurgery**: Many integration tests - ensure file paths are correct
4. **TestAlignment**: File I/O heavy - may need cleanup fixtures
5. **Language model tests**: May have external dependencies

## Validation

After converting each file:
1. Run `python -m pytest tests/test_<name>.py -v`
2. Ensure all tests pass
3. Check for any deprecation warnings
4. Verify test coverage is maintained
