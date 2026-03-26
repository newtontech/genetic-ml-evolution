"""
SQLite-based Cache System for Architecture Performance

This module provides a caching mechanism to avoid re-evaluating the same
architecture configurations during the evolutionary process.

Features:
- SQLite backend for persistent storage
- JSON serialization for architecture configurations
- Fast lookup by architecture hash
- Support for multiple performance metrics
- Cache statistics and management
"""

import sqlite3
import json
import hashlib
import os
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class ArchitectureCache:
    """
    SQLite-based cache for storing and retrieving architecture performance results.
    
    The cache stores architecture configurations along with their evaluation metrics,
    avoiding redundant evaluations during the evolutionary process.
    
    Attributes:
        db_path (str): Path to the SQLite database file
        conn (sqlite3.Connection): Database connection
        cache_hits (int): Number of cache hits
        cache_misses (int): Number of cache misses
    
    Example:
        >>> cache = ArchitectureCache("cache.db")
        >>> arch = {"type": "transformer", "num_layers": 6}
        >>> cache.store(arch, {"accuracy": 0.85, "loss": 0.15})
        >>> result = cache.lookup(arch)
        >>> print(result["accuracy"])
        0.85
    """
    
    def __init__(self, db_path: str = "architecture_cache.db"):
        """
        Initialize the cache system.
        
        Args:
            db_path: Path to SQLite database file. Defaults to "architecture_cache.db"
        """
        self.db_path = db_path
        self.conn = None
        self.cache_hits = 0
        self.cache_misses = 0
        self._initialize_database()
    
    def _initialize_database(self) -> None:
        """Create database tables if they don't exist."""
        # Ensure directory exists
        db_dir = os.path.dirname(self.db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)
        
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        
        cursor = self.conn.cursor()
        
        # Main cache table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS architecture_cache (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                architecture_hash TEXT UNIQUE NOT NULL,
                architecture_json TEXT NOT NULL,
                architecture_type TEXT,
                performance_metrics TEXT NOT NULL,
                evaluation_time REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                access_count INTEGER DEFAULT 1
            )
        ''')
        
        # Index for fast hash lookups
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_architecture_hash 
            ON architecture_cache(architecture_hash)
        ''')
        
        # Index for type-based queries
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_architecture_type 
            ON architecture_cache(architecture_type)
        ''')
        
        # Statistics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS cache_statistics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                total_entries INTEGER,
                cache_hits INTEGER,
                cache_misses INTEGER,
                hit_rate REAL
            )
        ''')
        
        self.conn.commit()
        logger.info(f"Cache database initialized at {self.db_path}")
    
    def _compute_hash(self, architecture: Dict[str, Any]) -> str:
        """
        Compute a deterministic hash for an architecture configuration.
        
        Args:
            architecture: Architecture configuration dictionary
            
        Returns:
            SHA256 hash string
        """
        # Sort keys for deterministic ordering
        arch_str = json.dumps(architecture, sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(arch_str.encode('utf-8')).hexdigest()
    
    def store(
        self, 
        architecture: Dict[str, Any], 
        metrics: Dict[str, float],
        evaluation_time: Optional[float] = None
    ) -> bool:
        """
        Store an architecture and its performance metrics in the cache.
        
        Args:
            architecture: Architecture configuration dictionary
            metrics: Performance metrics (e.g., {"accuracy": 0.85, "loss": 0.15})
            evaluation_time: Time taken for evaluation (optional)
            
        Returns:
            True if stored successfully, False if architecture already exists
            
        Raises:
            ValueError: If architecture or metrics are invalid
        """
        if not isinstance(architecture, dict):
            raise ValueError("Architecture must be a dictionary")
        if not isinstance(metrics, dict):
            raise ValueError("Metrics must be a dictionary")
        
        arch_hash = self._compute_hash(architecture)
        arch_json = json.dumps(architecture, sort_keys=True)
        arch_type = architecture.get("type", "unknown")
        metrics_json = json.dumps(metrics, sort_keys=True)
        
        cursor = self.conn.cursor()
        
        try:
            cursor.execute('''
                INSERT INTO architecture_cache 
                (architecture_hash, architecture_json, architecture_type, 
                 performance_metrics, evaluation_time)
                VALUES (?, ?, ?, ?, ?)
            ''', (arch_hash, arch_json, arch_type, metrics_json, evaluation_time))
            
            self.conn.commit()
            logger.debug(f"Stored architecture with hash {arch_hash[:16]}...")
            return True
            
        except sqlite3.IntegrityError:
            # Architecture already exists
            logger.debug(f"Architecture with hash {arch_hash[:16]}... already exists")
            return False
    
    def lookup(self, architecture: Dict[str, Any]) -> Optional[Dict[str, float]]:
        """
        Look up performance metrics for an architecture.
        
        Args:
            architecture: Architecture configuration dictionary
            
        Returns:
            Performance metrics dictionary if found, None otherwise
        """
        arch_hash = self._compute_hash(architecture)
        
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT performance_metrics, id 
            FROM architecture_cache 
            WHERE architecture_hash = ?
        ''', (arch_hash,))
        
        row = cursor.fetchone()
        
        if row:
            self.cache_hits += 1
            # Update access statistics
            cursor.execute('''
                UPDATE architecture_cache 
                SET last_accessed = CURRENT_TIMESTAMP,
                    access_count = access_count + 1
                WHERE id = ?
            ''', (row['id'],))
            self.conn.commit()
            
            logger.debug(f"Cache hit for architecture {arch_hash[:16]}...")
            return json.loads(row['performance_metrics'])
        else:
            self.cache_misses += 1
            logger.debug(f"Cache miss for architecture {arch_hash[:16]}...")
            return None
    
    def exists(self, architecture: Dict[str, Any]) -> bool:
        """
        Check if an architecture exists in the cache.
        
        Args:
            architecture: Architecture configuration dictionary
            
        Returns:
            True if architecture exists, False otherwise
        """
        arch_hash = self._compute_hash(architecture)
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT 1 FROM architecture_cache WHERE architecture_hash = ?
        ''', (arch_hash,))
        return cursor.fetchone() is not None
    
    def get_by_type(
        self, 
        arch_type: str, 
        limit: Optional[int] = None
    ) -> List[Tuple[Dict[str, Any], Dict[str, float]]]:
        """
        Get all cached architectures of a specific type.
        
        Args:
            arch_type: Architecture type (e.g., "transformer", "cnn", "multimodal")
            limit: Maximum number of results to return
            
        Returns:
            List of (architecture, metrics) tuples
        """
        cursor = self.conn.cursor()
        
        if limit:
            cursor.execute('''
                SELECT architecture_json, performance_metrics 
                FROM architecture_cache 
                WHERE architecture_type = ?
                ORDER BY created_at DESC
                LIMIT ?
            ''', (arch_type, limit))
        else:
            cursor.execute('''
                SELECT architecture_json, performance_metrics 
                FROM architecture_cache 
                WHERE architecture_type = ?
                ORDER BY created_at DESC
            ''', (arch_type,))
        
        results = []
        for row in cursor.fetchall():
            arch = json.loads(row['architecture_json'])
            metrics = json.loads(row['performance_metrics'])
            results.append((arch, metrics))
        
        return results
    
    def get_top_performing(
        self,
        metric: str = "accuracy",
        limit: int = 10,
        arch_type: Optional[str] = None
    ) -> List[Tuple[Dict[str, Any], Dict[str, float]]]:
        """
        Get top performing architectures based on a specific metric.

        Args:
            metric: Performance metric to sort by (default: "accuracy")
            limit: Number of top results to return
            arch_type: Filter by architecture type (optional)

        Returns:
            List of (architecture, metrics_dict) tuples
        """
        cursor = self.conn.cursor()

        # Note: This is a simplified approach. For better performance,
        # consider storing metrics as separate columns for sorting.
        if arch_type:
            cursor.execute('''
                SELECT architecture_json, performance_metrics
                FROM architecture_cache
                WHERE architecture_type = ?
            ''', (arch_type,))
        else:
            cursor.execute('''
                SELECT architecture_json, performance_metrics
                FROM architecture_cache
            ''')

        results = []
        for row in cursor.fetchall():
            arch = json.loads(row['architecture_json'])
            metrics = json.loads(row['performance_metrics'])
            if metric in metrics:
                results.append((arch, metrics))

        # Sort by metric (descending for accuracy, ascending for loss)
        reverse = metric in ["accuracy", "f1", "precision", "recall"]
        results.sort(key=lambda x: x[1][metric], reverse=reverse)

        return results[:limit]

    def delete(self, architecture: Dict[str, Any]) -> bool:
        """
        Delete an architecture from the cache.

        Args:
            architecture: Architecture configuration dictionary

        Returns:
            True if deleted successfully, False if not found
        """
        arch_hash = self._compute_hash(architecture)
        cursor = self.conn.cursor()

        cursor.execute('''
            SELECT id FROM architecture_cache WHERE architecture_hash = ?
        ''', (arch_hash,))
        row = cursor.fetchone()

        if row:
            cursor.execute('''
                DELETE FROM architecture_cache WHERE id = ?
            ''', (row['id'],))
            self.conn.commit()
            logger.debug(f"Deleted architecture with hash {arch_hash[:16]}...")
            return True
        else:
            logger.debug(f"Architecture with hash {arch_hash[:16]}... not found")
            return False
    
    def clear(self) -> int:
        """
        Clear all entries from the cache.
        
        Returns:
            Number of entries removed
        """
        cursor = self.conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM architecture_cache')
        count = cursor.fetchone()[0]
        
        cursor.execute('DELETE FROM architecture_cache')
        self.conn.commit()
        
        logger.info(f"Cleared {count} entries from cache")
        return count
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary containing cache statistics
        """
        cursor = self.conn.cursor()
        
        # Total entries
        cursor.execute('SELECT COUNT(*) FROM architecture_cache')
        total_entries = cursor.fetchone()[0]
        
        # Entries by type
        cursor.execute('''
            SELECT architecture_type, COUNT(*) as count 
            FROM architecture_cache 
            GROUP BY architecture_type
        ''')
        by_type = {row['architecture_type']: row['count'] for row in cursor.fetchall()}
        
        # Average evaluation time
        cursor.execute('SELECT AVG(evaluation_time) FROM architecture_cache')
        avg_eval_time = cursor.fetchone()[0]
        
        # Hit rate
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = (self.cache_hits / total_requests * 100) if total_requests > 0 else 0.0
        
        return {
            "total_entries": total_entries,
            "entries_by_type": by_type,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate_percent": round(hit_rate, 2),
            "average_evaluation_time": round(avg_eval_time, 4) if avg_eval_time else None
        }
    
    def export_to_json(self, output_path: str) -> None:
        """
        Export cache contents to a JSON file.
        
        Args:
            output_path: Path to output JSON file
        """
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT architecture_json, performance_metrics, evaluation_time, created_at
            FROM architecture_cache
            ORDER BY created_at DESC
        ''')
        
        data = []
        for row in cursor.fetchall():
            data.append({
                "architecture": json.loads(row['architecture_json']),
                "metrics": json.loads(row['performance_metrics']),
                "evaluation_time": row['evaluation_time'],
                "created_at": row['created_at']
            })
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Exported {len(data)} cache entries to {output_path}")
    
    def import_from_json(self, input_path: str, overwrite: bool = False) -> int:
        """
        Import cache contents from a JSON file.
        
        Args:
            input_path: Path to input JSON file
            overwrite: If True, clear existing cache before import
            
        Returns:
            Number of entries imported
        """
        with open(input_path, 'r') as f:
            data = json.load(f)
        
        if overwrite:
            self.clear()
        
        imported = 0
        for entry in data:
            if self.store(
                entry['architecture'], 
                entry['metrics'],
                entry.get('evaluation_time')
            ):
                imported += 1
        
        logger.info(f"Imported {imported} entries from {input_path}")
        return imported
    
    def close(self) -> None:
        """Close the database connection."""
        if self.conn:
            self.conn.close()
            logger.info("Cache database connection closed")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
    
    def __len__(self) -> int:
        """Return number of cached entries."""
        cursor = self.conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM architecture_cache')
        return cursor.fetchone()[0]
    
    def __repr__(self) -> str:
        """String representation."""
        stats = self.get_statistics()
        return (
            f"ArchitectureCache(db_path='{self.db_path}', "
            f"entries={stats['total_entries']}, "
            f"hit_rate={stats['hit_rate_percent']}%)"
        )
