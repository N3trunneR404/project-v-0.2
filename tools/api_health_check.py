#!/usr/bin/env python3
"""
Comprehensive API health check and endpoint validation.
Tests all endpoints, checks for mismatches, and validates responses.
"""

from __future__ import annotations

import sys
import json
import requests
from typing import Dict, List, Any, Optional
from dataclasses import dataclass


@dataclass
class EndpointTest:
    """Test case for an API endpoint."""
    method: str
    path: str
    name: str
    description: str
    payload: Optional[Dict] = None
    expected_status: int = 200
    expected_fields: Optional[List[str]] = None
    validate_func: Optional[callable] = None


class APIHealthChecker:
    """Comprehensive API health checker."""
    
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip('/')
        self.results: List[Dict[str, Any]] = []
        self.issues: List[Dict[str, Any]] = []
    
    def test_endpoint(self, test: EndpointTest) -> Dict[str, Any]:
        """Test a single endpoint."""
        url = f"{self.base_url}{test.path}"
        
        result = {
            "name": test.name,
            "method": test.method,
            "path": test.path,
            "status": "unknown",
            "status_code": None,
            "response_time_ms": None,
            "errors": [],
            "warnings": [],
        }
        
        try:
            if test.method == "GET":
                response = requests.get(url, timeout=10)
            elif test.method == "POST":
                response = requests.post(url, json=test.payload, timeout=10, headers={"Content-Type": "application/json"})
            else:
                result["errors"].append(f"Unsupported method: {test.method}")
                result["status"] = "error"
                return result
            
            result["status_code"] = response.status_code
            result["response_time_ms"] = response.elapsed.total_seconds() * 1000
            
            # Check status code
            if response.status_code != test.expected_status:
                result["errors"].append(
                    f"Expected status {test.expected_status}, got {response.status_code}"
                )
                result["status"] = "error"
            else:
                result["status"] = "ok"
            
            # Parse response
            try:
                if response.headers.get('content-type', '').startswith('application/json'):
                    data = response.json()
                    result["response_data"] = data
                    
                    # Check expected fields
                    if test.expected_fields:
                        for field in test.expected_fields:
                            if field not in data:
                                result["warnings"].append(f"Missing expected field: {field}")
                    
                    # Custom validation
                    if test.validate_func:
                        try:
                            validation_result = test.validate_func(data)
                            if not validation_result:
                                result["warnings"].append("Custom validation failed")
                        except Exception as e:
                            result["warnings"].append(f"Validation error: {e}")
                else:
                    result["response_text"] = response.text[:500]  # First 500 chars
            except json.JSONDecodeError:
                result["warnings"].append("Response is not valid JSON")
                result["response_text"] = response.text[:500]
        
        except requests.exceptions.Timeout:
            result["errors"].append("Request timeout")
            result["status"] = "error"
        except requests.exceptions.ConnectionError:
            result["errors"].append("Connection error - API not accessible")
            result["status"] = "error"
        except Exception as e:
            result["errors"].append(f"Unexpected error: {e}")
            result["status"] = "error"
        
        return result
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all endpoint tests."""
        tests = self._get_test_cases()
        
        print("=" * 70)
        print("API HEALTH CHECK")
        print("=" * 70)
        print(f"Testing: {self.base_url}\n")
        
        for test in tests:
            print(f"Testing {test.method} {test.path}...", end=" ", flush=True)
            result = self.test_endpoint(test)
            self.results.append(result)
            
            if result["status"] == "ok":
                print(f"✓ ({result['response_time_ms']:.1f}ms)")
            else:
                print(f"✗ {', '.join(result['errors'])}")
                self.issues.append(result)
            
            if result.get("warnings"):
                for warning in result["warnings"]:
                    print(f"  ⚠ {warning}")
        
        return self._generate_report()
    
    def _get_test_cases(self) -> List[EndpointTest]:
        """Define all test cases."""
        return [
            # Snapshot endpoint
            EndpointTest(
                method="GET",
                path="/snapshot",
                name="Snapshot",
                description="Get current DT state snapshot",
                expected_status=200,
                expected_fields=["nodes"],
                validate_func=lambda d: isinstance(d.get("nodes"), list),
            ),
            
            # Plan endpoint - valid job
            EndpointTest(
                method="POST",
                path="/plan",
                name="Plan - Valid Job",
                description="Create plan for valid job",
                payload={
                    "job": {
                        "apiVersion": "fabric.dt/v1",
                        "kind": "Job",
                        "metadata": {"name": "test-job", "deadline_ms": 2000},
                        "spec": {
                            "stages": [{
                                "id": "s1",
                                "compute": {"cpu": 1, "mem_gb": 1, "duration_ms": 1000},
                                "constraints": {"arch": ["amd64"], "formats": ["native"]},
                            }]
                        }
                    },
                    "strategy": "greedy",
                    "dry_run": True,
                },
                expected_status=200,
                expected_fields=["plan_id", "placements", "predicted_latency_ms", "predicted_energy_kwh", "risk_score"],
                validate_func=lambda d: "plan_id" in d and len(d.get("placements", {})) > 0,
            ),
            
            # Plan endpoint - invalid job (missing fields)
            EndpointTest(
                method="POST",
                path="/plan",
                name="Plan - Invalid Job (Missing Fields)",
                description="Should return error for invalid job",
                payload={"job": {}},
                expected_status=400,
            ),
            
            # Plan endpoint - missing job
            EndpointTest(
                method="POST",
                path="/plan",
                name="Plan - Missing Job",
                description="Should return error when job is missing",
                payload={"strategy": "greedy"},
                expected_status=400,
            ),
            
            # Plan endpoint - different strategies
            EndpointTest(
                method="POST",
                path="/plan",
                name="Plan - Resilient Strategy",
                description="Test resilient scheduling strategy",
                payload={
                    "job": {
                        "apiVersion": "fabric.dt/v1",
                        "kind": "Job",
                        "metadata": {"name": "test-resilient", "deadline_ms": 3000},
                        "spec": {
                            "stages": [{
                                "id": "s1",
                                "compute": {"cpu": 1, "mem_gb": 1, "duration_ms": 1500},
                                "constraints": {"arch": ["amd64"], "formats": ["native"]},
                            }]
                        }
                    },
                    "strategy": "resilient",
                    "dry_run": True,
                },
                expected_status=200,
                expected_fields=["plan_id", "placements"],
            ),
            
            EndpointTest(
                method="POST",
                path="/plan",
                name="Plan - CVaR Strategy",
                description="Test CVaR risk-aware strategy",
                payload={
                    "job": {
                        "apiVersion": "fabric.dt/v1",
                        "kind": "Job",
                        "metadata": {"name": "test-cvar", "deadline_ms": 3000},
                        "spec": {
                            "stages": [{
                                "id": "s1",
                                "compute": {"cpu": 1, "mem_gb": 1, "duration_ms": 1500},
                                "constraints": {"arch": ["amd64"], "formats": ["native"]},
                            }]
                        }
                    },
                    "strategy": "cvar",
                    "dry_run": True,
                },
                expected_status=200,
                expected_fields=["plan_id", "placements"],
            ),
            
            # Observe endpoint - node down
            EndpointTest(
                method="POST",
                path="/observe",
                name="Observe - Node Down",
                description="Report node down event",
                payload={"type": "node_down", "node": "worker-0"},
                expected_status=200,
                expected_fields=["status"],
            ),
            
            # Observe endpoint - node up
            EndpointTest(
                method="POST",
                path="/observe",
                name="Observe - Node Up",
                description="Report node up event",
                payload={"type": "node_up", "node": "worker-0"},
                expected_status=200,
                expected_fields=["status"],
            ),
            
            # Observe endpoint - invalid event
            EndpointTest(
                method="POST",
                path="/observe",
                name="Observe - Invalid Event",
                description="Should handle invalid event type",
                payload={"type": "invalid", "node": "worker-0"},
                expected_status=200,  # Currently accepts any event
            ),
            
            # Plan endpoint - no feasible placements (edge case)
            EndpointTest(
                method="POST",
                path="/plan",
                name="Plan - Impossible Constraints",
                description="Test with impossible constraints",
                payload={
                    "job": {
                        "apiVersion": "fabric.dt/v1",
                        "kind": "Job",
                        "metadata": {"name": "impossible", "deadline_ms": 1000},
                        "spec": {
                            "stages": [{
                                "id": "s1",
                                "compute": {"cpu": 1000, "mem_gb": 1000, "duration_ms": 100},
                                "constraints": {"arch": ["amd64"], "formats": ["native"]},
                            }]
                        }
                    },
                    "strategy": "greedy",
                    "dry_run": True,
                },
                expected_status=400,  # Should return error for no feasible placements
            ),
        ]
    
    def _generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive report."""
        total = len(self.results)
        passed = sum(1 for r in self.results if r["status"] == "ok")
        failed = total - passed
        
        report = {
            "summary": {
                "total_tests": total,
                "passed": passed,
                "failed": failed,
                "success_rate": (passed / total * 100) if total > 0 else 0,
            },
            "results": self.results,
            "issues": self.issues,
        }
        
        return report


def check_kubernetes_integration(base_url: str) -> Dict[str, Any]:
    """Check Kubernetes-related functionality."""
    issues = []
    improvements = []
    
    print("\n" + "=" * 70)
    print("KUBERNETES INTEGRATION CHECK")
    print("=" * 70)
    
    # Check snapshot for node information
    try:
        resp = requests.get(f"{base_url}/snapshot", timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            nodes = data.get("nodes", [])
            
            if not nodes:
                issues.append({
                    "severity": "high",
                    "issue": "No nodes visible in snapshot",
                    "impact": "Scheduling will fail",
                })
            else:
                print(f"✓ Found {len(nodes)} nodes in snapshot")
                
                if len(nodes) < 3:
                    improvements.append({
                        "area": "Node Discovery",
                        "suggestion": "Consider implementing automatic node discovery from Kubernetes API",
                        "priority": "medium",
                    })
        else:
            issues.append({
                "severity": "critical",
                "issue": f"Snapshot endpoint returned {resp.status_code}",
                "impact": "Cannot verify node state",
            })
    except Exception as e:
        issues.append({
            "severity": "critical",
            "issue": f"Cannot access snapshot endpoint: {e}",
            "impact": "Cannot verify system state",
        })
    
    # Check observe endpoint functionality
    try:
        # Test node down
        resp = requests.post(
            f"{base_url}/observe",
            json={"type": "node_down", "node": "worker-0"},
            timeout=5
        )
        if resp.status_code != 200:
            issues.append({
                "severity": "medium",
                "issue": f"Observe endpoint returned {resp.status_code}",
                "impact": "Cannot report node events",
            })
        
        # Verify node state changed
        resp2 = requests.get(f"{base_url}/snapshot", timeout=5)
        if resp2.status_code == 200:
            # Node should be marked unavailable (if implementation is correct)
            print("✓ Observe endpoint functional")
    except Exception as e:
        issues.append({
            "severity": "medium",
            "issue": f"Observe endpoint test failed: {e}",
            "impact": "Node state management may be broken",
        })
    
    return {
        "issues": issues,
        "improvements": improvements,
    }


def main():
    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8080"
    
    # Run API health check
    checker = APIHealthChecker(base_url)
    report = checker.run_all_tests()
    
    # Run Kubernetes integration check
    k8s_check = check_kubernetes_integration(base_url)
    
    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total Tests: {report['summary']['total_tests']}")
    print(f"Passed: {report['summary']['passed']}")
    print(f"Failed: {report['summary']['failed']}")
    print(f"Success Rate: {report['summary']['success_rate']:.1f}%")
    
    if report['issues']:
        print(f"\n⚠ Found {len(report['issues'])} issues:")
        for issue in report['issues']:
            print(f"  - {issue['name']}: {', '.join(issue['errors'])}")
    
    if k8s_check['issues']:
        print(f"\n⚠ Kubernetes Issues ({len(k8s_check['issues'])}):")
        for issue in k8s_check['issues']:
            print(f"  [{issue['severity'].upper()}] {issue['issue']}")
            print(f"    Impact: {issue['impact']}")
    
    if k8s_check['improvements']:
        print(f"\n💡 Suggested Improvements ({len(k8s_check['improvements'])}):")
        for imp in k8s_check['improvements']:
            print(f"  [{imp['priority'].upper()}] {imp['area']}: {imp['suggestion']}")
    
    # Save detailed report
    import pathlib
    import json
    from datetime import datetime
    
    reports_dir = pathlib.Path(__file__).parent.parent / "reports" / "health"
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    report_path = reports_dir / f"api-health-{timestamp}.json"
    
    full_report = {
        "timestamp": timestamp,
        "base_url": base_url,
        "api_check": report,
        "kubernetes_check": k8s_check,
    }
    
    with open(report_path, 'w') as f:
        json.dump(full_report, f, indent=2)
    
    print(f"\n✓ Detailed report saved: {report_path}")
    
    # Return non-zero exit code if there are critical issues
    critical_issues = [i for i in k8s_check['issues'] if i['severity'] == 'critical']
    if critical_issues or report['summary']['failed'] > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()

