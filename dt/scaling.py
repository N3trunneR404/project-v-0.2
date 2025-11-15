"""Resource scaling framework for simulation."""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ResourceScaler:
	"""
	Scales resources for simulation purposes.
	
	Default scale: 1:100 (1 simulated CPU = 0.01 real cores)
	"""
	
	cpu_scale: float = 0.01  # 1 simulated CPU = 0.01 real cores (1:100)
	memory_scale: float = 0.01  # 1 simulated GB = 0.01 real GB (1:100)
	duration_scale: float = 1.0  # Duration typically not scaled (1:1)
	
	def __init__(
		self,
		cpu_scale: float = 0.01,
		memory_scale: float = 0.01,
		duration_scale: float = 1.0
	) -> None:
		"""
		Initialize resource scaler.
		
		Args:
			cpu_scale: Scale factor for CPU (default 0.01 = 1:100)
			memory_scale: Scale factor for memory (default 0.01 = 1:100)
			duration_scale: Scale factor for duration (default 1.0 = no scaling)
		"""
		self.cpu_scale = cpu_scale
		self.memory_scale = memory_scale
		self.duration_scale = duration_scale
		logger.info(
			f"ResourceScaler initialized: CPU={cpu_scale:.4f}, "
			f"Memory={memory_scale:.4f}, Duration={duration_scale:.4f}"
		)
	
	def scale_cpu(self, units: int) -> float:
		"""
		Scale CPU units to real cores.
		
		Args:
			units: Simulated CPU units
			
		Returns:
			Real CPU cores
		"""
		return float(units) * self.cpu_scale
	
	def scale_memory(self, units_gb: int) -> float:
		"""
		Scale memory units to real GB.
		
		Args:
			units_gb: Simulated memory in GB
			
		Returns:
			Real memory in GB
		"""
		return float(units_gb) * self.memory_scale
	
	def scale_duration(self, duration_ms: int) -> float:
		"""
		Scale duration (typically no scaling, but configurable).
		
		Args:
			duration_ms: Simulated duration in milliseconds
			
		Returns:
			Real duration in milliseconds
		"""
		return float(duration_ms) * self.duration_scale
	
	def unscale_cpu(self, real_cores: float) -> int:
		"""
		Convert real CPU cores back to simulated units.
		
		Args:
			real_cores: Real CPU cores
			
		Returns:
			Simulated CPU units
		"""
		return int(real_cores / self.cpu_scale)
	
	def unscale_memory(self, real_gb: float) -> int:
		"""
		Convert real memory back to simulated units.
		
		Args:
			real_gb: Real memory in GB
			
		Returns:
			Simulated memory in GB
		"""
		return int(real_gb / self.memory_scale)
	
	def get_scale_factor(self) -> float:
		"""
		Get the primary scale factor (CPU scale).
		
		Returns:
			Scale factor (e.g., 0.01 for 1:100)
		"""
		return self.cpu_scale
	
	def to_dict(self) -> dict:
		"""Export scaling configuration as dictionary."""
		return {
			'cpu_scale': self.cpu_scale,
			'memory_scale': self.memory_scale,
			'duration_scale': self.duration_scale,
			'scale_ratio': f"1:{int(1.0 / self.cpu_scale)}"
		}


# Default scaler instance
DEFAULT_SCALER = ResourceScaler()

