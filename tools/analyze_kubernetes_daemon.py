#!/usr/bin/env python3
"""
Analyze Kubernetes daemon/actuator for improvements and issues.
"""

from __future__ import annotations

import sys
import pathlib
from typing import List, Dict, Any

# Add project root
root = pathlib.Path(__file__).resolve().parents[1]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

# Note: kubernetes module may not be available in local environment
# from dt.actuator import Actuator
# from dt.state import DTState, Node, HardwareSpec, NodeRuntime, KubernetesInfo, Job, JobStage, StageCompute, StageConstraints, PlacementDecision


def analyze_actuator() -> List[Dict[str, Any]]:
    """Analyze the Actuator class for issues and improvements."""
    issues = []
    improvements = []
    
    print("=" * 70)
    print("ACTUATOR ANALYSIS")
    print("=" * 70)
    
    # Issue 1: Pod creation is stubbed
    issues.append({
        "severity": "high",
        "issue": "Pod creation is stubbed - no actual Kubernetes pods are created",
        "location": "dt/actuator.py:submit_plan()",
        "impact": "Plans are not actually executed in Kubernetes",
        "code": """
        if isinstance(pod, dict):
            # For now, just log - actual pod creation would need proper V1Pod object
            # self.core.create_namespaced_pod(namespace=self.namespace, body=pod)
            pass
        """,
    })
    
    improvements.append({
        "priority": "high",
        "area": "Pod Creation",
        "suggestion": "Implement proper V1Pod object creation from placement decisions",
        "details": """
        - Convert dict to V1Pod using kubernetes.client.V1Pod
        - Set proper container specs, resource requests/limits
        - Add node selectors based on placement decision
        - Handle image selection based on exec_format (native/qemu/wasm)
        """,
    })
    
    # Issue 2: Cordon/Uncordon creates new API client each time
    issues.append({
        "severity": "medium",
        "issue": "cordon_node/uncordon_node create new API client instead of using self.core",
        "location": "dt/actuator.py:cordon_node() and uncordon_node()",
        "impact": "Inefficient and inconsistent API client usage",
        "code": """
        def cordon_node(self, node_name: str) -> None:
            body = {"spec": {"unschedulable": True}}
            client.CoreV1Api().patch_node(node_name, body)  # Creates new client
        """,
    })
    
    improvements.append({
        "priority": "medium",
        "area": "API Client Management",
        "suggestion": "Use self.core consistently for all Kubernetes API calls",
        "details": "Replace client.CoreV1Api() with self.core in cordon_node and uncordon_node",
    })
    
    # Issue 3: No error handling
    issues.append({
        "severity": "medium",
        "issue": "No error handling for Kubernetes API calls",
        "location": "dt/actuator.py:submit_plan(), cordon_node(), uncordon_node()",
        "impact": "Failures are not caught or reported",
        "details": "Should handle ApiException and other Kubernetes client errors",
    })
    
    improvements.append({
        "priority": "medium",
        "area": "Error Handling",
        "suggestion": "Add try-except blocks for Kubernetes API calls",
        "details": """
        - Catch kubernetes.client.exceptions.ApiException
        - Log errors appropriately
        - Return meaningful error messages
        - Consider retry logic for transient failures
        """,
    })
    
    # Issue 4: No validation of node existence
    improvements.append({
        "priority": "low",
        "area": "Validation",
        "suggestion": "Validate node exists before cordoning/uncordoning",
        "details": "Check node exists in cluster before attempting to modify it",
    })
    
    # Issue 5: No plan status tracking
    improvements.append({
        "priority": "medium",
        "area": "Plan Management",
        "suggestion": "Track plan execution status",
        "details": """
        - Store plan status (pending, running, completed, failed)
        - Query pod status to update plan state
        - Provide API endpoint to query plan status
        """,
    })
    
    # Issue 6: No resource cleanup
    improvements.append({
        "priority": "low",
        "area": "Resource Management",
        "suggestion": "Implement plan cancellation and cleanup",
        "details": "Add method to delete pods associated with a plan",
    })
    
    return issues, improvements


def analyze_state_management() -> List[Dict[str, Any]]:
    """Analyze state management for Kubernetes integration."""
    issues = []
    improvements = []
    
    print("\n" + "=" * 70)
    print("STATE MANAGEMENT ANALYSIS")
    print("=" * 70)
    
    # Issue: No automatic node discovery
    issues.append({
        "severity": "high",
        "issue": "Nodes are manually seeded, no automatic discovery from Kubernetes",
        "location": "app.py:seed_state()",
        "impact": "DT state may be out of sync with actual cluster state",
    })
    
    improvements.append({
        "priority": "high",
        "area": "Node Discovery",
        "suggestion": "Implement automatic node discovery from Kubernetes API",
        "details": """
        - Watch Kubernetes nodes using informer/watch
        - Automatically add/update nodes in DT state
        - Sync node labels, allocatable resources, conditions
        - Handle node removal
        """,
    })
    
    # Issue: No pod status tracking
    improvements.append({
        "priority": "medium",
        "area": "Pod Tracking",
        "suggestion": "Track pod status for submitted plans",
        "details": """
        - Watch pods created by actuator
        - Update plan status based on pod status
        - Handle pod failures and retries
        """,
    })
    
    # Issue: mark_node_availability may not work for full DTState
    issues.append({
        "severity": "medium",
        "issue": "mark_node_availability may not properly update full DTState implementation",
        "location": "dt/state.py:mark_node_availability()",
        "impact": "Node availability changes may not persist correctly",
    })
    
    return issues, improvements


def generate_improvements_report(actuator_issues, actuator_improvements, 
                                 state_issues, state_improvements):
    """Generate comprehensive improvements report."""
    
    report = f"""# Kubernetes Daemon Improvements Analysis

## Summary

- **Total Issues Found:** {len(actuator_issues) + len(state_issues)}
- **Total Improvements Suggested:** {len(actuator_improvements) + len(state_improvements)}

---

## Actuator Issues

"""
    
    for i, issue in enumerate(actuator_issues, 1):
        report += f"""### Issue {i}: {issue['issue']}

**Severity:** {issue['severity'].upper()}  
**Location:** `{issue['location']}`  
**Impact:** {issue['impact']}

"""
        if 'code' in issue:
            report += f"**Code:**\n```python{issue['code']}\n```\n\n"
        if 'details' in issue:
            report += f"**Details:** {issue['details']}\n\n"
    
    report += "## Actuator Improvements\n\n"
    
    for i, imp in enumerate(actuator_improvements, 1):
        report += f"""### Improvement {i}: {imp['area']}

**Priority:** {imp['priority'].upper()}  
**Suggestion:** {imp['suggestion']}

**Details:**
{imp['details']}

"""
    
    report += "## State Management Issues\n\n"
    
    for i, issue in enumerate(state_issues, 1):
        report += f"""### Issue {i}: {issue['issue']}

**Severity:** {issue['severity'].upper()}  
**Location:** `{issue['location']}`  
**Impact:** {issue['impact']}

"""
    
    report += "## State Management Improvements\n\n"
    
    for i, imp in enumerate(state_improvements, 1):
        report += f"""### Improvement {i}: {imp['area']}

**Priority:** {imp['priority'].upper()}  
**Suggestion:** {imp['suggestion']}

**Details:**
{imp['details']}

"""
    
    report += """---

## Recommended Implementation Order

1. **High Priority:**
   - Fix pod creation stub (actual Kubernetes integration)
   - Implement automatic node discovery
   - Add error handling for API calls

2. **Medium Priority:**
   - Fix API client usage in cordon/uncordon
   - Add plan status tracking
   - Improve mark_node_availability for full DTState

3. **Low Priority:**
   - Add node validation
   - Implement resource cleanup
   - Add plan cancellation

---

*Report generated automatically*
"""
    
    return report


def main():
    actuator_issues, actuator_improvements = analyze_actuator()
    state_issues, state_improvements = analyze_state_management()
    
    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Actuator Issues: {len(actuator_issues)}")
    print(f"Actuator Improvements: {len(actuator_improvements)}")
    print(f"State Issues: {len(state_issues)}")
    print(f"State Improvements: {len(state_improvements)}")
    
    # Generate report
    reports_dir = root / "reports" / "health"
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    report_path = reports_dir / f"kubernetes-improvements-{timestamp}.md"
    
    report = generate_improvements_report(
        actuator_issues, actuator_improvements,
        state_issues, state_improvements
    )
    
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"\n✓ Report saved: {report_path}")


if __name__ == "__main__":
    main()

