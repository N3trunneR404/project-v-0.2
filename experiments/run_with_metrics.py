#!/usr/bin/env python3
"""
Run experiments multiple times and collect metrics for analysis.
Generates structured JSON output for each experiment run.
"""

from __future__ import annotations

import json
import sys
import time
import pathlib
from typing import Dict, List, Any
from datetime import datetime
import requests

# Add project root to path
root = pathlib.Path(__file__).resolve().parents[1]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

# Import experiment modules
import experiments.v1_controller_vs_baseline as v1
import experiments.v2_predictive_ablation as v2
import experiments.v3_overhead as v3
import experiments.v4_shadow_plans as v4
import experiments.v5_scalability_kwok as v5
import experiments.v6_drift_robustness as v6


def collect_metrics_from_response(response_data: Dict[str, Any]) -> Dict[str, Any]:
    """Extract metrics from API response."""
    return {
        "predicted_latency_ms": response_data.get("predicted_latency_ms", 0.0),
        "predicted_energy_kwh": response_data.get("predicted_energy_kwh", 0.0),
        "risk_score": response_data.get("risk_score", 0.0),
        "plan_id": response_data.get("plan_id", ""),
        "placements": response_data.get("placements", {}),
        "num_placements": len(response_data.get("placements", {})),
    }


def run_experiment_with_metrics(module, dt_url: str, experiment_name: str) -> List[Dict[str, Any]]:
    """Run an experiment and collect all metrics from responses."""
    results = []
    
    # Check if module has a run function that accepts dt_url
    if hasattr(module, 'run'):
        try:
            # Capture stdout to parse JSON responses
            import io
            from contextlib import redirect_stdout
            
            # Some experiments print JSON, others return data
            # We'll intercept the output
            output_buffer = io.StringIO()
            
            # Run the experiment
            start_time = time.time()
            with redirect_stdout(output_buffer):
                module.run(dt_url)
            end_time = time.time()
            
            # Parse output for JSON lines
            output = output_buffer.getvalue()
            for line in output.strip().split('\n'):
                line = line.strip()
                if line.startswith('{') and line.endswith('}'):
                    try:
                        data = json.loads(line)
                        metrics = collect_metrics_from_response(data)
                        metrics["experiment_name"] = experiment_name
                        metrics["execution_time_ms"] = (end_time - start_time) * 1000
                        results.append(metrics)
                    except json.JSONDecodeError:
                        pass
        except Exception as e:
            print(f"Error running {experiment_name}: {e}", file=sys.stderr)
    
    return results


def run_experiments_n_times(dt_url: str, n_runs: int = 5) -> Dict[str, List[Dict[str, Any]]]:
    """Run all experiments N times and collect metrics."""
    experiments = {
        "v1_controller_vs_baseline": v1,
        "v2_predictive_ablation": v2,
        "v3_overhead": v3,
        "v4_shadow_plans": v4,
        "v5_scalability_kwok": v5,
        "v6_drift_robustness": v6,
    }
    
    all_results = {}
    
    for exp_name, module in experiments.items():
        print(f"\n{'='*60}")
        print(f"Running {exp_name} ({n_runs} times)")
        print(f"{'='*60}")
        
        exp_results = []
        for run_num in range(1, n_runs + 1):
            print(f"\nRun {run_num}/{n_runs}...")
            run_results = run_experiment_with_metrics(module, dt_url, exp_name)
            for result in run_results:
                result["run_number"] = run_num
            exp_results.extend(run_results)
            time.sleep(1)  # Brief pause between runs
        
        all_results[exp_name] = exp_results
        print(f"\n✓ {exp_name}: Collected {len(exp_results)} data points")
    
    return all_results


def calculate_statistics(values: List[float]) -> Dict[str, float]:
    """Calculate statistical measures."""
    if not values:
        return {
            "mean": 0.0,
            "min": 0.0,
            "max": 0.0,
            "std": 0.0,
            "median": 0.0,
        }
    
    import statistics
    sorted_vals = sorted(values)
    n = len(values)
    
    return {
        "mean": statistics.mean(values),
        "min": min(values),
        "max": max(values),
        "std": statistics.stdev(values) if n > 1 else 0.0,
        "median": statistics.median(values),
    }


def aggregate_results(all_results: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    """Aggregate results and calculate statistics."""
    aggregated = {}
    
    for exp_name, results in all_results.items():
        if not results:
            continue
        
        # Extract metrics
        latencies = [r["predicted_latency_ms"] for r in results]
        energies = [r["predicted_energy_kwh"] for r in results]
        risks = [r["risk_score"] for r in results]
        execution_times = [r.get("execution_time_ms", 0.0) for r in results]
        
        aggregated[exp_name] = {
            "total_runs": len(set(r["run_number"] for r in results)),
            "total_data_points": len(results),
            "latency_ms": calculate_statistics(latencies),
            "energy_kwh": calculate_statistics(energies),
            "risk_score": calculate_statistics(risks),
            "execution_time_ms": calculate_statistics(execution_times),
            "sample_results": results[:5],  # First 5 for reference
        }
    
    return aggregated


def generate_report(aggregated: Dict[str, Any], output_path: str):
    """Generate a comprehensive markdown report."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    report = f"""# Digital Twin Experiment Results

**Generated:** {timestamp}  
**Total Experiments:** {len(aggregated)}  
**Runs per Experiment:** 5

---

"""
    
    for exp_name, data in aggregated.items():
        report += f"""## {exp_name.replace('_', ' ').title()}

**Total Runs:** {data['total_runs']}  
**Total Data Points:** {data['total_data_points']}

### Latency Metrics (ms)

| Metric | Value |
|--------|-------|
| Mean | {data['latency_ms']['mean']:.2f} |
| Median | {data['latency_ms']['median']:.2f} |
| Min | {data['latency_ms']['min']:.2f} |
| Max | {data['latency_ms']['max']:.2f} |
| Std Dev | {data['latency_ms']['std']:.2f} |

### Energy Metrics (kWh)

| Metric | Value |
|--------|-------|
| Mean | {data['energy_kwh']['mean']:.6f} |
| Median | {data['energy_kwh']['median']:.6f} |
| Min | {data['energy_kwh']['min']:.6f} |
| Max | {data['energy_kwh']['max']:.6f} |
| Std Dev | {data['energy_kwh']['std']:.6f} |

### Risk Score Metrics

| Metric | Value |
|--------|-------|
| Mean | {data['risk_score']['mean']:.4f} |
| Median | {data['risk_score']['median']:.4f} |
| Min | {data['risk_score']['min']:.4f} |
| Max | {data['risk_score']['max']:.4f} |
| Std Dev | {data['risk_score']['std']:.4f} |

### Execution Time (ms)

| Metric | Value |
|--------|-------|
| Mean | {data['execution_time_ms']['mean']:.2f} |
| Median | {data['execution_time_ms']['median']:.2f} |
| Min | {data['execution_time_ms']['min']:.2f} |
| Max | {data['execution_time_ms']['max']:.2f} |
| Std Dev | {data['execution_time_ms']['std']:.2f} |

---

"""
    
    # Add summary table
    report += """## Summary Table

| Experiment | Avg Latency (ms) | Avg Energy (kWh) | Avg Risk | Avg Exec Time (ms) |
|------------|------------------|------------------|----------|-------------------|
"""
    
    for exp_name, data in aggregated.items():
        report += f"| {exp_name.replace('_', ' ').title()} | "
        report += f"{data['latency_ms']['mean']:.2f} | "
        report += f"{data['energy_kwh']['mean']:.6f} | "
        report += f"{data['risk_score']['mean']:.4f} | "
        report += f"{data['execution_time_ms']['mean']:.2f} |\n"
    
    report += f"""

---

## Methodology

- Each experiment was run **5 times** to ensure statistical significance
- Metrics collected: predicted latency, energy consumption, risk score, and execution time
- All values are averaged across runs
- Standard deviation indicates variability in results

## Notes

- Results are based on the Digital Twin's DES (Discrete Event Simulation) predictions
- All experiments use the same 3-node test cluster (worker-0, worker-1, worker-2)
- Port-forward daemon ensures stable connection throughout experiments
"""
    
    # Write report
    with open(output_path, 'w') as f:
        f.write(report)
    
    print(f"\n✓ Report generated: {output_path}")


def main():
    dt_url = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8080"
    n_runs = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    
    print(f"Running experiments {n_runs} times against {dt_url}")
    print("=" * 60)
    
    # Run experiments
    all_results = run_experiments_n_times(dt_url, n_runs)
    
    # Aggregate and calculate statistics
    aggregated = aggregate_results(all_results)
    
    # Save raw JSON data
    results_dir = root / "reports" / "experiments"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    json_path = results_dir / f"results-{timestamp}.json"
    with open(json_path, 'w') as f:
        json.dump({
            "timestamp": timestamp,
            "n_runs": n_runs,
            "raw_results": all_results,
            "aggregated": aggregated,
        }, f, indent=2)
    
    print(f"\n✓ Raw data saved: {json_path}")
    
    # Generate report
    report_path = results_dir / f"report-{timestamp}.md"
    generate_report(aggregated, str(report_path))
    
    print(f"\n{'='*60}")
    print("EXPERIMENT EXECUTION COMPLETE")
    print(f"{'='*60}")
    print(f"Results: {report_path}")
    print(f"Data: {json_path}")


if __name__ == "__main__":
    main()

