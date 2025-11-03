#!/usr/bin/env python3
"""
Simple test script to verify that the --length_init argument works correctly.
"""

import sys
from learnMSA.run.args import parse_args

def test_single_length():
    """Test with a single length value"""
    print("Test 1: Single length value")
    sys.argv = ['learnMSA', '-i', 'input.fasta', '-o', 'output.a2m', '--length_init', '5']
    parser = parse_args("test")
    args = parser.parse_args()
    assert args.length_init == [5], f"Expected [5], got {args.length_init}"
    print(f"  ✓ Single value: {args.length_init}")

def test_multiple_lengths():
    """Test with multiple length values"""
    print("Test 2: Multiple length values")
    sys.argv = ['learnMSA', '-i', 'input.fasta', '-o', 'output.a2m', '--length_init', '2', '3', '4', '5']
    parser = parse_args("test")
    args = parser.parse_args()
    assert args.length_init == [2, 3, 4, 5], f"Expected [2, 3, 4, 5], got {args.length_init}"
    print(f"  ✓ Multiple values: {args.length_init}")

def test_default_none():
    """Test that default is None when not specified"""
    print("Test 3: Default value (not specified)")
    sys.argv = ['learnMSA', '-i', 'input.fasta', '-o', 'output.a2m']
    parser = parse_args("test")
    args = parser.parse_args()
    assert args.length_init is None, f"Expected None, got {args.length_init}"
    print(f"  ✓ Default value: {args.length_init}")

def test_large_values():
    """Test with larger length values"""
    print("Test 4: Large length values")
    sys.argv = ['learnMSA', '-i', 'input.fasta', '-o', 'output.a2m', '--length_init', '10', '20', '30']
    parser = parse_args("test")
    args = parser.parse_args()
    assert args.length_init == [10, 20, 30], f"Expected [10, 20, 30], got {args.length_init}"
    print(f"  ✓ Large values: {args.length_init}")

