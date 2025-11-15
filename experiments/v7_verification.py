"""V7: Prediction Verification Experiment

This experiment validates the accuracy of Digital Twin predictions
by comparing predicted metrics against observed metrics from actual executions.
"""

from __future__ import annotations

import sys
import time
import json
from pathlib import Path
from typing import Dict, Any, List

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import requests
from dt.verification import PredictionVerifier, VerificationResult
from dt.state import ObservedMetrics, Plan


def create_test_job(
    name: str,
    origin_cluster: str = "edge-microdc",
    origin_node: str = None
) -> Dict[str, Any]:
    """Create a test job with origin context."""
    return {
        "metadata": {
            "name": name,
            "deadline_ms": 10000,
            "origin": {
                "cluster": origin_cluster,
                "node": origin_node,
            }
        },
        "spec": {
            "stages": [
                {
                    "id": "s1",
                    "compute": {
                        "cpu": 2,
                        "mem_gb": 1,
                        "duration_ms": 2000,
                        "workload_type": "cpu_bound"
                    },
                    "constraints": {
                        "arch": ["amd64"],
                        "formats": ["native"]
                    }
                }
            ]
        }
    }


def run_verification_experiment(
    api_url: str = "http://127.0.0.1:8080",
    n_jobs: int = 10
) -> List[Dict[str, Any]]:
    """
    Run verification experiment: submit jobs, collect predictions and observations.
    
    Args:
        api_url: DT API URL
        n_jobs: Number of jobs to submit
        
    Returns:
        List of verification results
    """
    verifier = PredictionVerifier(
        output_dir="reports/verification",
        latency_threshold_pct=10.0,
        energy_threshold_pct=20.0,
    )
    
    results: List[Dict[str, Any]] = []
    
    print(f"=== V7 Verification Experiment ===")
    print(f"Submitting {n_jobs} jobs for verification...")
    print()
    
    for i in range(n_jobs):
        job_name = f"verify-job-{i+1}"
        job_spec = create_test_job(job_name, origin_cluster="edge-microdc")
        
        # Submit plan
        print(f"Job {i+1}/{n_jobs}: {job_name}")
        try:
            plan_response = requests.post(
                f"{api_url}/plan",
                json={"job": job_spec, "strategy": "greedy", "dry_run": False},
                timeout=30
            )
            plan_response.raise_for_status()
            plan_data = plan_response.json()
            
            plan_id = plan_data.get("plan_id")
            predicted_latency = plan_data.get("predicted_latency_ms", 0.0)
            predicted_energy = plan_data.get("predicted_energy_kwh", 0.0)
            risk_score = plan_data.get("risk_score", 0.0)
            
            print(f"  Plan ID: {plan_id}")
            print(f"  Predicted: latency={predicted_latency:.2f}ms, energy={predicted_energy:.6f}kWh")
            
            # Wait for pod completion (simplified - in real implementation, use telemetry collector)
            print(f"  Waiting for execution...")
            time.sleep(5)  # Simplified wait
            
            # Get observed metrics (in real implementation, from telemetry collector)
            verify_response = requests.get(
                f"{api_url}/plan/{plan_id}/verify",
                timeout=10
            )
            
            if verify_response.status_code == 200:
                verify_data = verify_response.json()
                observed = verify_data.get("observed", {})
                
                observed_metrics = ObservedMetrics(
                    latency_ms=observed.get("latency_ms", 0.0),
                    cpu_util=observed.get("cpu_util", 0.0),
                    mem_peak_gb=observed.get("mem_peak_gb", 0.0),
                    energy_kwh=observed.get("energy_kwh", 0.0),
                    completed_at=observed.get("completed_at"),
                )
                
                plan = Plan(
                    plan_id=plan_id,
                    job_name=job_name,
                    placements={},  # Would be populated from plan_data
                    predicted_latency_ms=predicted_latency,
                    predicted_energy_kwh=predicted_energy,
                    risk_score=risk_score,
                )
                
                # Verify
                verification_result = verifier.verify(plan, observed_metrics)
                
                results.append({
                    "plan_id": plan_id,
                    "job_name": job_name,
                    "verification": {
                        "latency_error_pct": verification_result.latency_error_rel,
                        "energy_error_pct": verification_result.energy_error_rel,
                        "latency_within_threshold": verification_result.latency_within_threshold,
                        "energy_within_threshold": verification_result.energy_within_threshold,
                        "acceptable": verification_result.is_acceptable(),
                    },
                    "predicted": {
                        "latency_ms": predicted_latency,
                        "energy_kwh": predicted_energy,
                    },
                    "observed": {
                        "latency_ms": observed_metrics.latency_ms,
                        "energy_kwh": observed_metrics.energy_kwh,
                    },
                })
                
                status = "✓ ACCEPTABLE" if verification_result.is_acceptable() else "✗ OUT OF THRESHOLD"
                print(f"  {status}")
                print(f"    Latency error: {verification_result.latency_error_rel:.1f}%")
                print(f"    Energy error: {verification_result.energy_error_rel:.1f}%")
            else:
                print(f"  ⚠ No observed metrics available (status: {verify_response.status_code})")
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
        
        print()
    
    # Compute aggregate statistics
    if results:
        verification_results = []
        for r in results:
            # Reconstruct VerificationResult for aggregate stats
            # (simplified - in real implementation, store full results)
            pass
        
        aggregate = verifier.compute_aggregate_stats([
            VerificationResult(
                plan_id=r["plan_id"],
                predicted_latency_ms=r["predicted"]["latency_ms"],
                observed_latency_ms=r["observed"]["latency_ms"],
                predicted_energy_kwh=r["predicted"]["energy_kwh"],
                observed_energy_kwh=r["observed"]["energy_kwh"],
                predicted_risk_score=0.0,
                latency_error_abs=abs(r["observed"]["latency_ms"] - r["predicted"]["latency_ms"]),
                latency_error_rel=r["verification"]["latency_error_pct"],
                energy_error_abs=abs(r["observed"]["energy_kwh"] - r["predicted"]["energy_kwh"]),
                energy_error_rel=r["verification"]["energy_error_pct"],
                latency_within_threshold=r["verification"]["latency_within_threshold"],
                energy_within_threshold=r["verification"]["energy_within_threshold"],
            )
            for r in results
        ])
        
        print("=== Aggregate Statistics ===")
        print(f"Total jobs: {aggregate['total_plans']}")
        print(f"Acceptable: {aggregate['acceptable_plans']} ({aggregate['acceptance_rate']:.1f}%)")
        print(f"Latency error: {aggregate['latency_error_mean_pct']:.1f}% ± {aggregate['latency_error_std_pct']:.1f}%")
        print(f"Energy error: {aggregate['energy_error_mean_pct']:.1f}% ± {aggregate['energy_error_std_pct']:.1f}%")
        print()
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="V7: Prediction Verification Experiment")
    parser.add_argument("--api-url", default="http://127.0.0.1:8080", help="DT API URL")
    parser.add_argument("--n-jobs", type=int, default=10, help="Number of jobs to submit")
    parser.add_argument("--output", help="Output JSON file")
    
    args = parser.parse_args()
    
    results = run_verification_experiment(api_url=args.api_url, n_jobs=args.n_jobs)
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output}")

