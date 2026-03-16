"""
Cache System - 缓存系统
避免重复评估相同架构
"""

import json
import hashlib
from typing import Dict, Optional, Any, Union
from pathlib import Path


class CacheSystem:
    """架构评估缓存系统"""
    
    def __init__(self, cache_dir: str = ".cache/genetic_ml"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.memory_cache: Dict[str, float] = {}
        self.hit_count = 0
        self.miss_count = 0
    
    def _get_cache_key(self, architecture: Dict[str, Any]) -> str:
        """生成架构的缓存键"""
        arch_str = json.dumps(architecture, sort_keys=True)
        return hashlib.md5(arch_str.encode()).hexdigest()
    
    def _get_cache_file(self, key: str) -> Path:
        """获取缓存文件路径"""
        return self.cache_dir / f"{key}.json"
    
    def get(self, architecture: Dict[str, Any]) -> Optional[float]:
        """从缓存获取架构评估结果"""
        key = self._get_cache_key(architecture)
        
        # 先检查内存缓存
        if key in self.memory_cache:
            self.hit_count += 1
            return self.memory_cache[key]
        
        # 检查磁盘缓存
        cache_file = self._get_cache_file(key)
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                    fitness = data.get('fitness')
                    if fitness is not None:
                        self.memory_cache[key] = fitness
                        self.hit_count += 1
                        return fitness
            except (json.JSONDecodeError, IOError):
                pass
        
        self.miss_count += 1
        return None
    
    def put(self, architecture: Dict[str, Any], fitness: float) -> None:
        """将评估结果存入缓存"""
        key = self._get_cache_key(architecture)
        
        # 存入内存缓存
        self.memory_cache[key] = fitness
        
        # 存入磁盘缓存
        cache_file = self._get_cache_file(key)
        try:
            with open(cache_file, 'w') as f:
                json.dump({
                    'architecture': architecture,
                    'fitness': fitness,
                }, f)
        except IOError:
            pass
    
    def get_stats(self) -> Dict[str, Union[int, float]]:
        """获取缓存统计信息"""
        return {
            'hits': self.hit_count,
            'misses': self.miss_count,
            'total': self.hit_count + self.miss_count,
            'hit_rate': self.hit_count / max(1, self.hit_count + self.miss_count),
            'memory_entries': len(self.memory_cache),
        }
    
    def clear(self) -> None:
        """清空缓存"""
        self.memory_cache.clear()
        self.hit_count = 0
        self.miss_count = 0
        
        # 删除磁盘缓存文件
        for cache_file in self.cache_dir.glob("*.json"):
            try:
                cache_file.unlink()
            except IOError:
                pass
