"""
ConnectTel Production Verification Suite - Master Test Runner
Executes both Production QA (4 tests) and REST API Automated Suite (10 tests).
"""

import os
import sys
import subprocess

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    prod_dir = os.path.join(current_dir, "production")
    
    print("=" * 65)
    print("  RUNNING CONNECTTEL PRODUCTION VERIFICATION HARNESS")
    print("=" * 65)
    
    # Run test_production.py
    cmd_prod = [sys.executable, os.path.join(prod_dir, "test_production.py")]
    res_prod = subprocess.run(cmd_prod, cwd=prod_dir)
    
    # Run test_api.py
    cmd_api = [sys.executable, os.path.join(prod_dir, "test_api.py")]
    res_api = subprocess.run(cmd_api, cwd=prod_dir)
    
    if res_prod.returncode == 0 and res_api.returncode == 0:
        print("\n" + "=" * 65)
        print("  ALL TESTS PASSED WITH 100% SUCCESS (0 FAILURES)")
        print("=" * 65)
        sys.exit(0)
    else:
        sys.exit(1)
