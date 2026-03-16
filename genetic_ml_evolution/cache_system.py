"""
Cache System - SQLite 缓存系统
避免重复评估相同架构，使用 SQLite 数据库持久化存储
"""

import json
import hashlib
import sqlite3
from typing import Dict, Optional, Any, Union, List
from pathlib import Path
from datetime import datetime
import threading


class CacheSystem:
    """架构评估缓存系统 - SQLite 实现
    
    使用 SQLite 数据库存储架构评估结果，提供高效的查询和持久化能力。
    
    Attributes:
        db_path: SQLite 数据库文件路径
        memory_cache: 内存缓存，用于加速热点数据访问
        hit_count: 缓存命中次数
        miss_count: 缓存未命中次数
        _lock: 线程锁，确保并发安全
    """
    
    def __init__(self, cache_dir: str = ".cache/genetic_ml"):
        """初始化缓存系统
        
        Args:
            cache_dir: 缓存目录，SQLite 数据库将存储在此目录
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # SQLite 数据库文件路径
        self.db_path = self.cache_dir / "architecture_cache.db"
        
        # 内存缓存（热点数据）
        self.memory_cache: Dict[str, Dict[str, Any]] = {}
        
        # 统计信息
        self.hit_count = 0
        self.miss_count = 0
        
        # 线程锁（SQLite 连接不是线程安全的）
        self._lock = threading.Lock()
        
        # 初始化数据库
        self._init_database()
    
    def _get_connection(self) -> sqlite3.Connection:
        """获取数据库连接
        
        Returns:
            SQLite 数据库连接对象
        """
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row  # 使用 Row 对象访问结果
        return conn
    
    def _init_database(self) -> None:
        """初始化数据库表结构
        
        创建 architecture_cache 表，用于存储架构评估结果。
        表结构：
        - cache_key: 缓存键（架构哈希），主键
        - architecture: 架构配置（JSON 字符串）
        - fitness: 适应度分数
        - accuracy: 准确率
        - loss: 损失值
        - training_time: 训练时间（秒）
        - memory_usage: 显存使用（MB）
        - created_at: 创建时间
        - updated_at: 更新时间
        """
        with self._lock:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS architecture_cache (
                        cache_key TEXT PRIMARY KEY,
                        architecture TEXT NOT NULL,
                        fitness REAL NOT NULL,
                        accuracy REAL,
                        loss REAL,
                        training_time REAL,
                        memory_usage REAL,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    )
                ''')
                
                # 创建索引以加速查询
                cursor.execute('''
                    CREATE INDEX IF NOT EXISTS idx_fitness 
                    ON architecture_cache(fitness DESC)
                ''')
                
                cursor.execute('''
                    CREATE INDEX IF NOT EXISTS idx_created_at 
                    ON architecture_cache(created_at DESC)
                ''')
                
                conn.commit()
            finally:
                conn.close()
    
    def _get_cache_key(self, architecture: Dict[str, Any]) -> str:
        """生成架构的缓存键
        
        使用 MD5 哈希算法将架构配置转换为唯一的缓存键。
        
        Args:
            architecture: 架构配置字典
            
        Returns:
            32 字符的 MD5 哈希字符串
        """
        arch_str = json.dumps(architecture, sort_keys=True)
        return hashlib.md5(arch_str.encode()).hexdigest()
    
    def get(self, architecture: Dict[str, Any]) -> Optional[float]:
        """从缓存获取架构评估结果（适应度）
        
        优先从内存缓存读取，若未命中则从 SQLite 数据库读取。
        
        Args:
            architecture: 架构配置字典
            
        Returns:
            适应度分数（0-100），若缓存未命中返回 None
        """
        key = self._get_cache_key(architecture)
        
        # 1. 检查内存缓存
        if key in self.memory_cache:
            self.hit_count += 1
            return self.memory_cache[key].get('fitness')
        
        # 2. 检查 SQLite 数据库
        with self._lock:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute(
                    'SELECT fitness, accuracy, loss, training_time, memory_usage FROM architecture_cache WHERE cache_key = ?',
                    (key,)
                )
                row = cursor.fetchone()
                
                if row:
                    # 存入内存缓存
                    self.memory_cache[key] = {
                        'fitness': row['fitness'],
                        'accuracy': row['accuracy'],
                        'loss': row['loss'],
                        'training_time': row['training_time'],
                        'memory_usage': row['memory_usage']
                    }
                    self.hit_count += 1
                    return row['fitness']
            finally:
                conn.close()
        
        # 3. 缓存未命中
        self.miss_count += 1
        return None
    
    def get_detailed(self, architecture: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """从缓存获取详细的评估结果
        
        Args:
            architecture: 架构配置字典
            
        Returns:
            包含所有评估指标的字典，若缓存未命中返回 None
        """
        key = self._get_cache_key(architecture)
        
        # 1. 检查内存缓存
        if key in self.memory_cache:
            self.hit_count += 1
            return self.memory_cache[key]
        
        # 2. 检查 SQLite 数据库
        with self._lock:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute(
                    '''SELECT fitness, accuracy, loss, training_time, memory_usage 
                       FROM architecture_cache WHERE cache_key = ?''',
                    (key,)
                )
                row = cursor.fetchone()
                
                if row:
                    result = {
                        'fitness': row['fitness'],
                        'accuracy': row['accuracy'],
                        'loss': row['loss'],
                        'training_time': row['training_time'],
                        'memory_usage': row['memory_usage']
                    }
                    # 存入内存缓存
                    self.memory_cache[key] = result
                    self.hit_count += 1
                    return result
            finally:
                conn.close()
        
        self.miss_count += 1
        return None
    
    def put(
        self, 
        architecture: Dict[str, Any], 
        fitness: float,
        accuracy: Optional[float] = None,
        loss: Optional[float] = None,
        training_time: Optional[float] = None,
        memory_usage: Optional[float] = None
    ) -> None:
        """将评估结果存入缓存
        
        同时更新内存缓存和 SQLite 数据库。
        
        Args:
            architecture: 架构配置字典
            fitness: 适应度分数（必需）
            accuracy: 准确率（可选）
            loss: 损失值（可选）
            training_time: 训练时间（秒，可选）
            memory_usage: 显存使用（MB，可选）
        """
        key = self._get_cache_key(architecture)
        now = datetime.now().isoformat()
        
        # 准备缓存数据
        cache_data = {
            'fitness': fitness,
            'accuracy': accuracy,
            'loss': loss,
            'training_time': training_time,
            'memory_usage': memory_usage
        }
        
        # 1. 更新内存缓存
        self.memory_cache[key] = cache_data
        
        # 2. 更新 SQLite 数据库
        with self._lock:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO architecture_cache 
                    (cache_key, architecture, fitness, accuracy, loss, training_time, memory_usage, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    key,
                    json.dumps(architecture, sort_keys=True),
                    fitness,
                    accuracy,
                    loss,
                    training_time,
                    memory_usage,
                    now,
                    now
                ))
                conn.commit()
            finally:
                conn.close()
    
    def get_top_performers(self, limit: int = 10) -> List[Dict[str, Any]]:
        """获取表现最好的 N 个架构
        
        Args:
            limit: 返回的架构数量
            
        Returns:
            架构列表，每个元素包含架构配置和评估指标
        """
        with self._lock:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT architecture, fitness, accuracy, loss, training_time, memory_usage, created_at
                    FROM architecture_cache
                    ORDER BY fitness DESC
                    LIMIT ?
                ''', (limit,))
                
                results = []
                for row in cursor.fetchall():
                    results.append({
                        'architecture': json.loads(row['architecture']),
                        'fitness': row['fitness'],
                        'accuracy': row['accuracy'],
                        'loss': row['loss'],
                        'training_time': row['training_time'],
                        'memory_usage': row['memory_usage'],
                        'created_at': row['created_at']
                    })
                return results
            finally:
                conn.close()
    
    def get_stats(self) -> Dict[str, Union[int, float]]:
        """获取缓存统计信息
        
        Returns:
            包含缓存统计信息的字典：
            - hits: 命中次数
            - misses: 未命中次数
            - total: 总查询次数
            - hit_rate: 命中率
            - memory_entries: 内存缓存条目数
            - db_entries: 数据库条目数
            - db_size_mb: 数据库大小（MB）
        """
        stats = {
            'hits': self.hit_count,
            'misses': self.miss_count,
            'total': self.hit_count + self.miss_count,
            'hit_rate': self.hit_count / max(1, self.hit_count + self.miss_count),
            'memory_entries': len(self.memory_cache),
            'db_entries': 0,
            'db_size_mb': 0.0
        }
        
        # 获取数据库统计信息
        with self._lock:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                
                # 数据库条目数
                cursor.execute('SELECT COUNT(*) as count FROM architecture_cache')
                stats['db_entries'] = cursor.fetchone()['count']
                
                # 数据库大小
                if self.db_path.exists():
                    stats['db_size_mb'] = self.db_path.stat().st_size / (1024 * 1024)
            finally:
                conn.close()
        
        return stats
    
    def clear(self) -> None:
        """清空缓存
        
        删除所有内存缓存和数据库记录。
        """
        # 清空内存缓存
        self.memory_cache.clear()
        self.hit_count = 0
        self.miss_count = 0
        
        # 清空数据库
        with self._lock:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute('DELETE FROM architecture_cache')
                conn.commit()
            finally:
                conn.close()
    
    def vacuum(self) -> None:
        """清理数据库碎片，释放空间
        
        SQLite 的 VACUUM 命令会重建数据库文件，删除已删除记录占用的空间。
        """
        with self._lock:
            conn = self._get_connection()
            try:
                conn.execute('VACUUM')
                conn.commit()
            finally:
                conn.close()
    
    def export_to_json(self, output_path: str) -> None:
        """导出缓存到 JSON 文件
        
        Args:
            output_path: 输出文件路径
        """
        with self._lock:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT cache_key, architecture, fitness, accuracy, loss, training_time, memory_usage, created_at
                    FROM architecture_cache
                    ORDER BY fitness DESC
                ''')
                
                data = []
                for row in cursor.fetchall():
                    data.append({
                        'cache_key': row['cache_key'],
                        'architecture': json.loads(row['architecture']),
                        'fitness': row['fitness'],
                        'accuracy': row['accuracy'],
                        'loss': row['loss'],
                        'training_time': row['training_time'],
                        'memory_usage': row['memory_usage'],
                        'created_at': row['created_at']
                    })
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
            finally:
                conn.close()
    
    def import_from_json(self, input_path: str, overwrite: bool = False) -> int:
        """从 JSON 文件导入缓存
        
        Args:
            input_path: 输入文件路径
            overwrite: 是否覆盖现有缓存
            
        Returns:
            导入的条目数量
        """
        if overwrite:
            self.clear()
        
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        count = 0
        for item in data:
            self.put(
                architecture=item['architecture'],
                fitness=item['fitness'],
                accuracy=item.get('accuracy'),
                loss=item.get('loss'),
                training_time=item.get('training_time'),
                memory_usage=item.get('memory_usage')
            )
            count += 1
        
        return count
    
    def __len__(self) -> int:
        """返回缓存中的条目数"""
        stats = self.get_stats()
        return stats['db_entries']
    
    def __repr__(self) -> str:
        """返回缓存系统的字符串表示"""
        stats = self.get_stats()
        return (
            f"CacheSystem(db_entries={stats['db_entries']}, "
            f"memory_entries={stats['memory_entries']}, "
            f"hit_rate={stats['hit_rate']:.2%})"
        )
