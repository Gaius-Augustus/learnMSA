"""Structural tokens predicted from sequence.

``prostt5`` runs ProstT5 (PyTorch, ``transformers``) to predict per-residue
3Di logits; ``io`` reads and writes them with numpy only, so learnMSA can use
the files with either backend installed. The command line entry point is
``learnMSA-3di`` (``predict_3di``).
"""
