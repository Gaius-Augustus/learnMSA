# Unittest to Pytest Conversion Status

## Overview
Total test classes: 16  
Completed: 16 ✅  
In Progress: 0 🔄  
Remaining: 0 ⏳

## Test Classes

### ✅ Completed

1. **TestDataset** → `test_dataset.py`
   - Status: ✅ Complete
   - Tests: 11
   - Complexity: Medium
   - Notes: Warning suppression for BioPython implemented

2. **TestMsaHmmCell** → `test_msa_hmm_cell.py`
   - Status: ✅ Complete
   - Tests: 16
   - Complexity: High
   - Notes: Fixtures for test data, parallel and non-parallel tests

3. **TestData** → `test_data.py`
   - Status: ✅ Complete
   - Tests: 1
   - Complexity: Low
   - Notes: Simple batch generation test

4. **TestPriors** → `test_priors.py`
   - Status: ✅ Complete
   - Tests: 1
   - Complexity: Low
   - Notes: Amino acid prior testing

5. **TestModelToFile** → `test_model_to_file.py`
   - Status: ✅ Complete
   - Tests: 1 (comprehensive)
   - Complexity: Medium
   - Notes: Model serialization and deserialization

6. **TestMvnMixture** → `test_mvn_mixture.py`
   - Status: ✅ Complete
   - Tests: 2
   - Complexity: Medium
   - Notes: Statistical distribution tests

7. **TestLanguageModelExtension** → `test_language_model_extension.py`
   - Status: ✅ Complete
   - Tests: 2
   - Complexity: Medium
   - Notes: Embedding cache and regularizer tests

8. **TestEmbeddingPretrainingDatapipeline** → `test_embedding_pretraining_datapipeline.py`
   - Status: ✅ Complete
   - Tests: 1
   - Complexity: Low
   - Notes: Column occupancy calculation

9. **TestPretrainingUtilities** → `test_pretraining_utilities.py`
   - Status: ✅ Complete
   - Tests: 5
   - Complexity: Medium
   - Notes: Masked loss functions and utilities

10. **TestAncProbs** → `test_anc_probs.py`
    - Status: ✅ Complete
    - Tests: 5
    - Complexity: High
    - Notes: Complex initialization, multiple helper methods

11. **TestModelSurgery** → `test_model_surgery.py`
    - Status: ✅ Complete
    - Tests: 9
    - Complexity: High
    - Notes: make_test_alignment helper method

12. **TestAlignment** → `test_alignment.py`
    - Status: ✅ Complete
    - Tests: 4
    - Complexity: High
    - Notes: EGF alignment benchmark test

13. **TestMSAHMM** → `test_msa_hmm.py`
    - Status: ✅ Complete
    - Tests: 8
    - Complexity: Very High
    - Notes: Massive test_viterbi (~500 lines), helper functions

14. **DirichletTest** → `test_dirichlet.py`
    - Status: ✅ Complete
    - Tests: 2
    - Complexity: Low
    - Notes: Dirichlet distribution PDF tests

15. **ConsoleTest** → `test_console.py`
    - Status: ✅ Complete
    - Tests: 1
    - Complexity: Low
    - Notes: Error handling with subprocess

16. **ClusteringTest** → `test_clustering.py`
    - Status: ✅ Complete
    - Tests: 1
    - Complexity: Low
    - Notes: Sequence weight computation

## 📋 Remaining Conversions

The remaining test classes need to be converted following the same pattern. Here's the breakdown:

### 4. TestMSAHMM (LARGE - Priority High)
- **Source:** Lines 461-1028 in UnitTests.py
- **Target:** tests/test_msa_hmm.py
- **Test methods:**
  - test_matrices
  - test_cell  
  - test_viterbi (VERY LARGE - ~500 lines)
  - test_parallel_viterbi
  - test_aligned_insertions
  - test_backward
  - test_posterior_state_probabilities (appears twice - likely a duplicate)
  - test_sequence_weights

- **Helper functions to convert:**
  - `string_to_one_hot(s)` (module-level, lines 456-458)
  - `get_all_seqs(data, num_models)` (module-level, lines 460-470)
  - `assert_vec(self, x, y)` (convert to module-level)

- **Complexity:** ⭐⭐⭐⭐⭐ (Very High - largest test class)

### 5. TestAncProbs
- **Source:** Lines 1029-1246 in UnitTests.py
- **Target:** tests/test_anc_probs.py
- **Test methods:**
  - test_paml_parsing
  - test_rate_matrices
  - test_anc_probs
  - test_encoder_model
  - test_transposed

- **__init__ data to convert to fixture:**
  - `self.paml_all = [Utility.LG_paml] + Utility.LG4X_paml`
  - `self.A = SequenceDataset.alphabet[:20]`

- **Helper methods to convert:**
  - `assert_vec`, `parse_a`, `assert_equilibrium`, `assert_symmetric`
  - `assert_rate_matrix`, `assert_anc_probs`, `assert_anc_probs_layer`
  - `get_test_configs`, `get_simple_seq`

- **Complexity:** ⭐⭐⭐⭐ (High)

### 6. TestModelSurgery
- **Source:** Lines 1269-1571 in UnitTests.py
- **Target:** tests/test_model_surgery.py
- **Test methods:**
  - test_insert_col
  - test_remove_col
  - test_concat_models
  - test_change_length
  - test_change_num_models
  - test_slicing
  - test_remove_gaps
  - test_mask_func
  - test_mask_alignment

- **Helper methods:**
  - `assert_vec`
  - `make_test_alignment` (creates test alignment data)

- **Complexity:** ⭐⭐⭐⭐ (High - many integration tests)

### 7. TestAlignment
- **Source:** Lines 1572-1767 in UnitTests.py
- **Target:** tests/test_alignment.py
- **Test methods:**
  - test_insert_gaps
  - test_decode_msa
  - test_decode_msa_lm
  - test_pairwise_alignment
  - test_guide_tree_fasta
  - test_guide_tree_stockholm
  - test_progressive_alignment

- **Helper methods:**
  - `assert_vec`

- **Complexity:** ⭐⭐⭐⭐ (High - file I/O, MSA operations)

### 8. TestPriors
- **Source:** Lines 1768-1784 in UnitTests.py
- **Target:** tests/test_priors.py
- **Test methods:**
  - test_dirichlet_mixture

- **Complexity:** ⭐ (Low - single simple test)

### 9. TestModelToFile
- **Source:** Lines 1785-1897 in UnitTests.py
- **Target:** tests/test_model_to_file.py
- **Test methods:**
  - test_model_to_file
  - test_alignment_model
  - test_reconstruct_model
  - test_load_weights_to_new_model
  - test_emission_only

- **Complexity:** ⭐⭐⭐ (Medium - file I/O, model serialization)

### 10. TestMvnMixture
- **Source:** Lines 1898-1975 in UnitTests.py
- **Target:** tests/test_mvn_mixture.py
- **Test methods:**
  - test_mvn_mixture
  - test_mvn_mixture_from_data
  - test_bayes_theorem
  - test_em

- **Complexity:** ⭐⭐⭐ (Medium - statistical tests)

### 11. TestLanguageModelExtension
- **Source:** Lines 1976-2030 in UnitTests.py
- **Target:** tests/test_language_model_extension.py
- **Test methods:**
  - test_full_pipeline
  - test_cache
  - test_cache_overwrite

- **Complexity:** ⭐⭐⭐ (Medium - integration tests)

### 12. TestEmbeddingPretrainingDatapipeline
- **Source:** Lines 2031-2038 in UnitTests.py
- **Target:** tests/test_embedding_pretraining_datapipeline.py
- **Test methods:**
  - test_datapipeline

- **Complexity:** ⭐ (Low - single test)

### 13. TestPretrainingUtilities
- **Source:** Lines 2039-end in UnitTests.py
- **Target:** tests/test_pretraining_utilities.py
- **Test methods:**
  - test_pretraining
  - test_mask_function

- **Complexity:** ⭐⭐ (Low-Medium)

## Conversion Checklist for Each File

When converting a test class, follow these steps:

- [ ] Create new file `tests/test_<name>.py`
- [ ] Add necessary imports (os, warnings if needed, numpy, pytest, tensorflow, learnMSA modules)
- [ ] Convert `__init__` data to pytest fixtures if present
- [ ] Convert helper methods (`self.assert_*`, etc.) to module-level functions
- [ ] Convert each test method:
  - [ ] Remove `self` parameter
  - [ ] Replace `self.assert*` with `assert` or `pytest` equivalents
  - [ ] Add fixtures as function parameters if needed
- [ ] Test the file: `python -m pytest tests/test_<name>.py -v`
- [ ] Verify all tests pass
- [ ] Update this status document

## Quick Reference: Common Conversions

```python
# Assertions
self.assertEqual(a, b)              → assert a == b
self.assertTrue(x)                  → assert x
self.assertFalse(x)                 → assert not x
self.assertIsNone(x)                → assert x is None
self.assertIsNotNone(x)             → assert x is not None
self.assertIn(a, b)                 → assert a in b
self.assertRaises(Exception)        → with pytest.raises(Exception):

# Keep as-is
np.testing.assert_almost_equal()    → np.testing.assert_almost_equal()
np.testing.assert_equal()           → np.testing.assert_equal()

# Class initialization
class Test(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data = value

→

@pytest.fixture
def test_data():
    return value

# Helper methods
class Test(unittest.TestCase):
    def helper(self, x):
        return x * 2

→

def helper(x):
    return x * 2
```

## Files Already in tests/ Directory

Check what already exists:
```bash
ls -la tests/
```

Current known files:
- test_dataset.py ✅
- test_msa_hmm_cell.py ✅
- test_data.py ✅

## Next Steps

1. Start with simpler test classes (TestPriors, TestEmbeddingPretrainingDatapipeline)
2. Move to medium complexity (TestModelToFile, TestMvnMixture, TestPretrainingUtilities)
3. Tackle high complexity classes (TestAncProbs, TestModelSurgery, TestAlignment, TestLanguageModelExtension)
4. Finally convert TestMSAHMM (largest and most complex)

## Validation

After all conversions:
```bash
# Run all tests
python -m pytest tests/ -v

# Check coverage if available
python -m pytest tests/ --cov=learnMSA --cov-report=html

# Run specific test file
python -m pytest tests/test_<name>.py -v
```
