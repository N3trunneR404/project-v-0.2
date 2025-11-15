"""Prediction verification module for Digital Twin."""

from __future__ import annotations

import csv
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from dt.state import ObservedMetrics, Plan

logger = logging.getLogger(__name__)


@dataclass
class VerificationResult:
	"""Result of prediction verification."""
	plan_id: str
	predicted_latency_ms: float
	observed_latency_ms: float
	predicted_energy_kwh: float
	observed_energy_kwh: float
	predicted_risk_score: float
	
	# Error metrics
	latency_error_abs: float
	latency_error_rel: float
	energy_error_abs: float
	energy_error_rel: float
	
	# Thresholds
	latency_threshold_pct: float = 10.0
	energy_threshold_pct: float = 20.0
	
	# Status
	latency_within_threshold: bool = False
	energy_within_threshold: bool = False
	
	def is_acceptable(self) -> bool:
		"""Check if prediction is within acceptable thresholds."""
		return self.latency_within_threshold and self.energy_within_threshold


class PredictionVerifier:
	"""
	Verifies predicted metrics against observed metrics.
	
	Configurable delta thresholds:
	- Latency: ±10% (default)
	- Energy/CPU: ±20% (default)
	- Risk: comparative ranking (not absolute)
	"""
	
	def __init__(
		self,
		output_dir: str = "reports/verification",
		latency_threshold_pct: float = 10.0,
		energy_threshold_pct: float = 20.0,
	) -> None:
		"""
		Initialize prediction verifier.
		
		Args:
			output_dir: Directory for CSV output files
			latency_threshold_pct: Acceptable latency error percentage (default 10%)
			energy_threshold_pct: Acceptable energy error percentage (default 20%)
		"""
		self.output_dir = Path(output_dir)
		self.output_dir.mkdir(parents=True, exist_ok=True)
		self.latency_threshold_pct = latency_threshold_pct
		self.energy_threshold_pct = energy_threshold_pct
		
		logger.info(
			f"PredictionVerifier initialized: "
			f"latency_threshold={latency_threshold_pct}%, "
			f"energy_threshold={energy_threshold_pct}%"
		)
	
	def verify(
		self,
		plan: Plan,
		observed: ObservedMetrics
	) -> VerificationResult:
		"""
		Verify predicted metrics against observed metrics.
		
		Args:
			plan: Plan with predicted metrics
			observed: Observed metrics from execution
			
		Returns:
			VerificationResult with error metrics and status
		"""
		# Calculate absolute errors
		latency_error_abs = abs(observed.latency_ms - plan.predicted_latency_ms)
		energy_error_abs = abs(observed.energy_kwh - plan.predicted_energy_kwh)
		
		# Calculate relative errors (percentage)
		latency_error_rel = (
			(latency_error_abs / plan.predicted_latency_ms * 100.0)
			if plan.predicted_latency_ms > 0 else 0.0
		)
		energy_error_rel = (
			(energy_error_abs / plan.predicted_energy_kwh * 100.0)
			if plan.predicted_energy_kwh > 0 else 0.0
		)
		
		# Check if within thresholds
		latency_within = latency_error_rel <= self.latency_threshold_pct
		energy_within = energy_error_rel <= self.energy_threshold_pct
		
		result = VerificationResult(
			plan_id=plan.plan_id,
			predicted_latency_ms=plan.predicted_latency_ms,
			observed_latency_ms=observed.latency_ms,
			predicted_energy_kwh=plan.predicted_energy_kwh,
			observed_energy_kwh=observed.energy_kwh,
			predicted_risk_score=plan.risk_score,
			latency_error_abs=latency_error_abs,
			latency_error_rel=latency_error_rel,
			energy_error_abs=energy_error_abs,
			energy_error_rel=energy_error_rel,
			latency_threshold_pct=self.latency_threshold_pct,
			energy_threshold_pct=self.energy_threshold_pct,
			latency_within_threshold=latency_within,
			energy_within_threshold=energy_within,
		)
		
		# Export to CSV
		self._export_to_csv(result)
		
		# Log result
		status = "✓ ACCEPTABLE" if result.is_acceptable() else "✗ OUT OF THRESHOLD"
		logger.info(
			f"Verification for plan {plan.plan_id}: {status}\n"
			f"  Latency: {plan.predicted_latency_ms:.2f}ms predicted, "
			f"{observed.latency_ms:.2f}ms observed "
			f"({latency_error_rel:.1f}% error, threshold: {self.latency_threshold_pct}%)\n"
			f"  Energy: {plan.predicted_energy_kwh:.6f}kWh predicted, "
			f"{observed.energy_kwh:.6f}kWh observed "
			f"({energy_error_rel:.1f}% error, threshold: {self.energy_threshold_pct}%)"
		)
		
		return result
	
	def _export_to_csv(self, result: VerificationResult) -> None:
		"""Export verification result to CSV file."""
		csv_path = self.output_dir / f"plan-{result.plan_id}-metrics.csv"
		
		# Check if file exists to determine if we need headers
		file_exists = csv_path.exists()
		
		with open(csv_path, 'a', newline='') as f:
			writer = csv.writer(f)
			
			# Write headers if new file
			if not file_exists:
				writer.writerow([
					'plan_id',
					'predicted_latency_ms',
					'observed_latency_ms',
					'latency_error_abs',
					'latency_error_rel_pct',
					'latency_within_threshold',
					'predicted_energy_kwh',
					'observed_energy_kwh',
					'energy_error_abs',
					'energy_error_rel_pct',
					'energy_within_threshold',
					'predicted_risk_score',
					'overall_acceptable',
				])
			
			# Write data row
			writer.writerow([
				result.plan_id,
				f"{result.predicted_latency_ms:.2f}",
				f"{result.observed_latency_ms:.2f}",
				f"{result.latency_error_abs:.2f}",
				f"{result.latency_error_rel:.2f}",
				result.latency_within_threshold,
				f"{result.predicted_energy_kwh:.6f}",
				f"{result.observed_energy_kwh:.6f}",
				f"{result.energy_error_abs:.6f}",
				f"{result.energy_error_rel:.2f}",
				result.energy_within_threshold,
				f"{result.predicted_risk_score:.4f}",
				result.is_acceptable(),
			])
		
		logger.debug(f"Exported verification result to {csv_path}")
	
	def compute_aggregate_stats(
		self,
		results: list[VerificationResult]
	) -> Dict[str, float]:
		"""
		Compute aggregate statistics from multiple verification results.
		
		Args:
			results: List of verification results
			
		Returns:
			Dictionary with aggregate statistics
		"""
		if not results:
			return {}
		
		latency_errors = [r.latency_error_rel for r in results]
		energy_errors = [r.energy_error_rel for r in results]
		
		def mean(values: list[float]) -> float:
			return sum(values) / len(values) if values else 0.0
		
		def std(values: list[float]) -> float:
			if not values:
				return 0.0
			avg = mean(values)
			variance = sum((x - avg) ** 2 for x in values) / len(values)
			return math.sqrt(variance)
		
		def rmse(predicted: list[float], observed: list[float]) -> float:
			if len(predicted) != len(observed):
				return 0.0
			squared_errors = [(p - o) ** 2 for p, o in zip(predicted, observed)]
			return math.sqrt(mean(squared_errors))
		
		latency_predicted = [r.predicted_latency_ms for r in results]
		latency_observed = [r.observed_latency_ms for r in results]
		energy_predicted = [r.predicted_energy_kwh for r in results]
		energy_observed = [r.observed_energy_kwh for r in results]
		
		acceptable_count = sum(1 for r in results if r.is_acceptable())
		
		return {
			'total_plans': len(results),
			'acceptable_plans': acceptable_count,
			'acceptance_rate': acceptable_count / len(results) * 100.0,
			'latency_error_mean_pct': mean(latency_errors),
			'latency_error_std_pct': std(latency_errors),
			'latency_rmse_ms': rmse(latency_predicted, latency_observed),
			'energy_error_mean_pct': mean(energy_errors),
			'energy_error_std_pct': std(energy_errors),
			'energy_rmse_kwh': rmse(energy_predicted, energy_observed),
		}

