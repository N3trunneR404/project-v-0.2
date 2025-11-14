#!/usr/bin/env python3
"""
Check for additional improvements and optimization opportunities.
"""

from __future__ import annotations

import sys
import pathlib
import ast
import re
from typing import List, Dict, Any

root = pathlib.Path(__file__).resolve().parents[1]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))


def check_code_quality() -> List[Dict[str, Any]]:
    """Check for code quality improvements."""
    improvements = []
    
    # Check actuator for resource requirements
    actuator_path = root / "dt" / "actuator.py"
    if actuator_path.exists():
        content = actuator_path.read_text()
        
        # Check if resource requirements use job stage compute
        if "compute.cpu" not in content and "compute.mem_gb" not in content:
            improvements.append({
                "file": "dt/actuator.py",
                "type": "enhancement",
                "priority": "medium",
                "issue": "Resource requirements use defaults, not job stage compute specs",
                "suggestion": "Pass JobStage compute requirements to pod_gen and use them for resource requests/limits",
                "location": "submit_plan() -> generate_pod_from_decision()",
            })
    
    # Check pod_gen for resource usage
    pod_gen_path = root / "k8s_executor" / "pod_gen.py"
    if pod_gen_path.exists():
        content = pod_gen_path.read_text()
        
        if "_get_resource_requirements(compute_cpu: int = 1" in content:
            improvements.append({
                "file": "k8s_executor/pod_gen.py",
                "type": "enhancement",
                "priority": "medium",
                "issue": "Resource requirements use defaults",
                "suggestion": "Accept JobStage compute requirements as parameters and use them",
                "location": "generate_pod_from_decision()",
            })
    
    return improvements


def check_api_improvements() -> List[Dict[str, Any]]:
    """Check for API improvements."""
    improvements = []
    
    api_path = root / "dt" / "api.py"
    if api_path.exists():
        content = api_path.read_text()
        
        # Check for plan status endpoint
        if "@app.get(\"/plan/" not in content and "@app.get(\"/plan/status" not in content:
            improvements.append({
                "file": "dt/api.py",
                "type": "feature",
                "priority": "medium",
                "issue": "No plan status query endpoint",
                "suggestion": "Add GET /plan/<plan_id>/status to query pod status for a plan",
                "benefit": "Allows clients to check if plans are executing successfully",
            })
        
        # Check for plan cancellation
        if "cancel" not in content.lower() or "@app.delete(\"/plan" not in content:
            improvements.append({
                "file": "dt/api.py",
                "type": "feature",
                "priority": "low",
                "issue": "No plan cancellation endpoint",
                "suggestion": "Add DELETE /plan/<plan_id> to cancel and clean up pods",
                "benefit": "Allows cancelling long-running or failed plans",
            })
    
    return improvements


def check_state_improvements() -> List[Dict[str, Any]]:
    """Check for state management improvements."""
    improvements = []
    
    # Check for automatic node discovery
    app_path = root / "app.py"
    if app_path.exists():
        content = app_path.read_text()
        
        if "seed_state" in content and "kubernetes" not in content.lower():
            improvements.append({
                "file": "app.py",
                "type": "feature",
                "priority": "high",
                "issue": "Nodes are manually seeded, no automatic discovery",
                "suggestion": "Implement Kubernetes node watcher to automatically discover and sync nodes",
                "benefit": "DT state stays in sync with actual cluster state",
            })
    
    return improvements


def check_performance_improvements() -> List[Dict[str, Any]]:
    """Check for performance improvements."""
    improvements = []
    
    improvements.append({
        "file": "dt/actuator.py",
        "type": "optimization",
        "priority": "low",
        "issue": "Pod creation is sequential",
        "suggestion": "Create pods in parallel using asyncio or threading",
        "benefit": "Faster plan execution for multi-stage jobs",
    })
    
    improvements.append({
        "file": "dt/api.py",
        "type": "optimization",
        "priority": "low",
        "issue": "No response caching",
        "suggestion": "Add caching for snapshot endpoint (state doesn't change frequently)",
        "benefit": "Reduced load for monitoring/UI clients",
    })
    
    return improvements


def check_security_improvements() -> List[Dict[str, Any]]:
    """Check for security improvements."""
    improvements = []
    
    improvements.append({
        "file": "dt/api.py",
        "type": "security",
        "priority": "medium",
        "issue": "No input validation for job specs",
        "suggestion": "Add validation for CPU/memory limits, deadline ranges, etc.",
        "benefit": "Prevent resource exhaustion attacks",
    })
    
    improvements.append({
        "file": "dt/actuator.py",
        "type": "security",
        "priority": "medium",
        "issue": "No RBAC validation",
        "suggestion": "Check service account permissions before attempting pod creation",
        "benefit": "Better error messages and fail-fast behavior",
    })
    
    return improvements


def main():
    print("=" * 70)
    print("IMPROVEMENT OPPORTUNITIES ANALYSIS")
    print("=" * 70)
    print()
    
    all_improvements = []
    all_improvements.extend(check_code_quality())
    all_improvements.extend(check_api_improvements())
    all_improvements.extend(check_state_improvements())
    all_improvements.extend(check_performance_improvements())
    all_improvements.extend(check_security_improvements())
    
    # Group by priority
    high_priority = [i for i in all_improvements if i["priority"] == "high"]
    medium_priority = [i for i in all_improvements if i["priority"] == "medium"]
    low_priority = [i for i in all_improvements if i["priority"] == "low"]
    
    print(f"Total Improvements Found: {len(all_improvements)}")
    print(f"  High Priority: {len(high_priority)}")
    print(f"  Medium Priority: {len(medium_priority)}")
    print(f"  Low Priority: {len(low_priority)}")
    print()
    
    if high_priority:
        print("=" * 70)
        print("HIGH PRIORITY IMPROVEMENTS")
        print("=" * 70)
        for i, imp in enumerate(high_priority, 1):
            print(f"\n{i}. {imp['issue']}")
            print(f"   File: {imp['file']}")
            print(f"   Type: {imp['type']}")
            print(f"   Suggestion: {imp['suggestion']}")
            if 'benefit' in imp:
                print(f"   Benefit: {imp['benefit']}")
    
    if medium_priority:
        print("\n" + "=" * 70)
        print("MEDIUM PRIORITY IMPROVEMENTS")
        print("=" * 70)
        for i, imp in enumerate(medium_priority, 1):
            print(f"\n{i}. {imp['issue']}")
            print(f"   File: {imp['file']}")
            print(f"   Type: {imp['type']}")
            print(f"   Suggestion: {imp['suggestion']}")
            if 'benefit' in imp:
                print(f"   Benefit: {imp['benefit']}")
    
    # Save report
    import json
    from datetime import datetime
    
    reports_dir = root / "reports" / "improvements"
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    report_path = reports_dir / f"improvements-{timestamp}.json"
    
    with open(report_path, 'w') as f:
        json.dump({
            "timestamp": timestamp,
            "total": len(all_improvements),
            "by_priority": {
                "high": len(high_priority),
                "medium": len(medium_priority),
                "low": len(low_priority),
            },
            "improvements": all_improvements,
        }, f, indent=2)
    
    print(f"\n✓ Report saved: {report_path}")


if __name__ == "__main__":
    main()

