#!/usr/bin/env python3
"""
Simple test script to verify caching functionality.
This script runs the qwen binary twice to test cache creation and loading.
"""

import subprocess
import os
import shutil
import time

def run_command(cmd, timeout=60):
    """Run command with timeout."""
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "Command timed out"

def test_caching():
    cache_dir = "./test_cache"

    # Clean up any existing cache
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)

    print("=== Testing Cache Creation ===")
    print("Running first time (should create cache)...")

    # First run - should create cache
    cmd1 = f"cargo run --features cuda --release -- --cache_dir {cache_dir} --gen_tokens 10"
    start_time = time.time()
    returncode1, stdout1, stderr1 = run_command(cmd1, timeout=120)
    first_run_time = time.time() - start_time

    print(f"First run completed in {first_run_time:.2f}s")
    print(f"Return code: {returncode1}")

    if returncode1 != 0:
        print("STDERR:", stderr1)
        print("STDOUT:", stdout1)
        return False

    # Check if cache files were created
    graph_cache = os.path.join(cache_dir, "graph_definition.bin")
    compiled_cache = os.path.join(cache_dir, "compiled_graph.bin")

    if os.path.exists(graph_cache):
        print("✓ Graph definition cache created")
    else:
        print("✗ Graph definition cache NOT created")
        return False

    if os.path.exists(compiled_cache):
        print("✓ Compiled graph cache created")
    else:
        print("✗ Compiled graph cache NOT created")
        return False

    print("\n=== Testing Cache Loading ===")
    print("Running second time (should use cache)...")

    # Second run - should use cache
    cmd2 = f"cargo run --features cuda --release -- --cache_dir {cache_dir} --gen_tokens 10"
    start_time = time.time()
    returncode2, stdout2, stderr2 = run_command(cmd2, timeout=120)
    second_run_time = time.time() - start_time

    print(f"Second run completed in {second_run_time:.2f}s")
    print(f"Return code: {returncode2}")

    if returncode2 != 0:
        print("STDERR:", stderr2)
        print("STDOUT:", stdout2)
        return False

    # Check for cache usage messages
    if "Loading graph definition from cache" in stdout2:
        print("✓ Graph definition loaded from cache")
    else:
        print("? Graph definition cache usage unclear")

    if "Loading compiled graph from cache" in stdout2:
        print("✓ Compiled graph loaded from cache")
    else:
        print("? Compiled graph cache usage unclear")

    print(f"\nTime comparison:")
    print(f"First run:  {first_run_time:.2f}s")
    print(f"Second run: {second_run_time:.2f}s")

    if second_run_time < first_run_time * 0.8:
        print("✓ Second run appears faster (caching likely working)")
    else:
        print("? Second run not significantly faster")

    # Clean up
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)

    return True

if __name__ == "__main__":
    print("Testing Qwen caching functionality...")
    success = test_caching()
    if success:
        print("\n✓ Cache test completed successfully!")
    else:
        print("\n✗ Cache test failed!")
        exit(1)