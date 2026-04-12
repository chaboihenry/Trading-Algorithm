import subprocess
import json
import os

UNIVERSE_PATH = "the_models/curated_universe.json"


def run_diagnostic():
    print("====== INITIATING FULL PIPELINE DIAGNOSTIC ======\n")

    try:
        # 1. Force a sequential run of the actual pipeline modules
        print("[1/3] Running Macro Data Update...")
        subprocess.run(["python", "-m", "the_utilities.fetch_macro_data"], check=True)

        print("\n[2/3] Running Cluster Discovery...")
        subprocess.run(["python", "-m", "the_research_node.m1_cluster_discovery"], check=True)

        print("\n[3/3] Running HRP Portfolio Allocator...")
        subprocess.run(["python", "-m", "the_research_node.m1_portfolio_allocator"], check=True)

    except subprocess.CalledProcessError as e:
        print(f"\n[CRITICAL FAILURE] Diagnostic aborted due to module crash: {e}")
        return

    # 2. Inspect the final payload
    print("\n====== DIAGNOSTIC REPORT ======")
    if not os.path.exists(UNIVERSE_PATH):
        print("[FAIL] curated_universe.json was not generated.")
        return

    with open(UNIVERSE_PATH, 'r') as f:
        data = json.load(f)

    baskets = data.get("baskets", {})
    if not baskets:
        print("[WARNING] No cointegrated baskets were found in today's data.")
        return

    print(f"Total Cointegrated Strategies Found: {len(baskets)}")
    print("HRP Capital Allocations:")

    total_allocation = 0.0
    for name, payload in baskets.items():
        allocation = payload.get('capital_allocation', 0.0)
        total_allocation += allocation
        print(f"  -> {name}: {allocation * 100:.2f}%")

    print(f"\nTotal Capital Allocated: {total_allocation * 100:.1f}%")
    if abs(total_allocation - 1.0) > 0.01:
        print("[FAIL] Allocations do not sum to 100%. Check HRP math.")
    else:
        print("[PASS] HRP Math Verified. Payload ready for ASUS Execution Node.")


if __name__ == "__main__":
    run_diagnostic()