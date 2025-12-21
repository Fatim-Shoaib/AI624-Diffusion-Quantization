#!/usr/bin/env python3
"""
=============================================================================
MASTER DIAGNOSIS RUNNER
=============================================================================

Runs all diagnostic scripts and generates a comprehensive report.

Usage:
    python scripts/run_all_diagnostics.py [--quick]
    
    --quick: Run only fast diagnostics (skip progressive layers test)
"""

import sys
import subprocess
import argparse
from pathlib import Path
from datetime import datetime


def run_script(script_path, description):
    """Run a diagnostic script and capture output."""
    print(f"\n{'='*60}")
    print(f"RUNNING: {description}")
    print(f"Script: {script_path}")
    print('='*60 + '\n')
    
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=False,
            text=True,
            cwd=Path(__file__).parent.parent
        )
        return result.returncode == 0
    except Exception as e:
        print(f"Error running {script_path}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true', 
                       help='Skip slow diagnostics')
    args = parser.parse_args()
    
    scripts_dir = Path(__file__).parent
    
    diagnostics = [
        # (script_name, description, is_slow)
        ("diag_02_verify_hessian.py", "Verify Hessian Matrix Computation", False),
        ("diag_06_weight_stats.py", "Analyze Weight Statistics", False),
        ("diag_08_inspect_model.py", "Inspect Quantized Model", False),
        ("diag_07_save_load.py", "Test Save/Load Corruption", False),
        ("diag_04_rtn_vs_gptq.py", "Compare RTN vs GPTQ", False),
        ("diag_05_layer_groups.py", "Test Layer Groups", False),
        ("diag_03_timestep_accumulation.py", "Cross-Timestep Error Accumulation", True),
        ("diag_01_progressive_layers.py", "Progressive Layer Quantization", True),
    ]
    
    print("="*60)
    print("COMPREHENSIVE QUANTIZATION DIAGNOSTICS")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    results = {}
    
    for script_name, description, is_slow in diagnostics:
        if args.quick and is_slow:
            print(f"\n[SKIPPED - slow] {description}")
            results[script_name] = "skipped"
            continue
        
        script_path = scripts_dir / script_name
        if not script_path.exists():
            print(f"\n[MISSING] {script_path}")
            results[script_name] = "missing"
            continue
        
        success = run_script(str(script_path), description)
        results[script_name] = "passed" if success else "failed"
    
    # Summary
    print("\n" + "="*60)
    print("DIAGNOSTICS SUMMARY")
    print("="*60)
    
    for script, status in results.items():
        status_icon = {"passed": "✅", "failed": "❌", "skipped": "⏭️", "missing": "❓"}
        print(f"  {status_icon.get(status, '?')} {script}: {status}")
    
    print("\n" + "="*60)
    print("NEXT STEPS")
    print("="*60)
    print("""
Check the output directories for detailed results:
  - diagnosis_progressive/     - Progressive layer quantization images
  - diagnosis_timesteps/       - Cross-timestep divergence analysis
  - diagnosis_rtn_vs_gptq/     - RTN vs GPTQ comparison
  - diagnosis_layer_groups/    - Layer group sensitivity analysis
  - diagnosis_weight_stats/    - Weight distribution analysis
  - diagnosis_save_load/       - Save/load corruption test
  - diagnosis_inspect_model/   - Actual quantized model inspection

Look for patterns in the results to identify the root cause.
""")


if __name__ == "__main__":
    main()
