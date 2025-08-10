"""Data backup and recovery system."""

import json
import logging
import shutil
import time
import hashlib
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Union
import pandas as pd
import duckdb

logger = logging.getLogger(__name__)


@dataclass
class BackupMetadata:
    """Metadata for a data backup."""
    backup_id: str
    timestamp: float
    source_info: Dict[str, Any]
    data_hash: str
    size_bytes: int
    format: str
    compression: Optional[str]
    description: Optional[str] = None


@dataclass
class RestorePoint:
    """Point-in-time data restore point."""
    restore_point_id: str
    timestamp: float
    backup_id: str
    operation_type: str
    affected_records: int
    metadata: Dict[str, Any]


class DataBackupManager:
    """Manages data backups and recovery operations."""
    
    def __init__(self, backup_dir: Union[str, Path] = "./backups"):
        """Initialize backup manager.
        
        Args:
            backup_dir: Directory to store backups
        """
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize metadata database
        self.metadata_db = self.backup_dir / "backup_metadata.db"
        self._init_metadata_db()
        
        # Backup settings
        self.max_backups = 50
        self.compression_enabled = True
        
        logger.info(f"Initialized backup manager with directory: {self.backup_dir}")
    
    def _init_metadata_db(self):
        """Initialize metadata database."""
        try:
            with duckdb.connect(str(self.metadata_db)) as conn:
                # Create backups table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS backups (
                        backup_id VARCHAR PRIMARY KEY,
                        timestamp DOUBLE,
                        source_info VARCHAR,
                        data_hash VARCHAR,
                        size_bytes BIGINT,
                        format VARCHAR,
                        compression VARCHAR,
                        description VARCHAR
                    )
                """)
                
                # Create restore points table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS restore_points (
                        restore_point_id VARCHAR PRIMARY KEY,
                        timestamp DOUBLE,
                        backup_id VARCHAR,
                        operation_type VARCHAR,
                        affected_records INTEGER,
                        metadata VARCHAR,
                        FOREIGN KEY (backup_id) REFERENCES backups (backup_id)
                    )
                """)
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Failed to initialize metadata database: {e}")
            raise
    
    def create_backup(
        self,
        data: pd.DataFrame,
        description: Optional[str] = None,
        format: str = "parquet"
    ) -> str:
        """Create a backup of the data.
        
        Args:
            data: DataFrame to backup
            description: Optional description
            format: Storage format (parquet, csv, json)
            
        Returns:
            Backup ID
        """
        try:
            # Generate backup ID
            timestamp = time.time()
            data_str = data.to_string()
            data_hash = hashlib.sha256(data_str.encode()).hexdigest()
            backup_id = f"backup_{int(timestamp)}_{data_hash[:8]}"
            
            # Create backup file path
            backup_file = self.backup_dir / f"{backup_id}.{format}"
            if self.compression_enabled and format == "parquet":
                backup_file = backup_file.with_suffix(".parquet.gz")
            
            # Save data
            if format == "parquet":
                if self.compression_enabled:
                    data.to_parquet(backup_file, compression='gzip')
                else:
                    data.to_parquet(backup_file)
            elif format == "csv":
                data.to_csv(backup_file, index=False)
            elif format == "json":
                data.to_json(backup_file, orient='records')
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            # Calculate file size
            size_bytes = backup_file.stat().st_size
            
            # Create metadata
            metadata = BackupMetadata(
                backup_id=backup_id,
                timestamp=timestamp,
                source_info={
                    "shape": data.shape,
                    "columns": list(data.columns),
                    "dtypes": {col: str(dtype) for col, dtype in data.dtypes.items()}
                },
                data_hash=data_hash,
                size_bytes=size_bytes,
                format=format,
                compression="gzip" if self.compression_enabled else None,
                description=description
            )
            
            # Store metadata
            self._store_backup_metadata(metadata)
            
            # Cleanup old backups
            self._cleanup_old_backups()
            
            logger.info(f"Created backup {backup_id} ({size_bytes / 1024:.1f} KB)")
            return backup_id
            
        except Exception as e:
            logger.error(f"Failed to create backup: {e}")
            raise
    
    def restore_backup(self, backup_id: str) -> pd.DataFrame:
        """Restore data from backup.
        
        Args:
            backup_id: ID of backup to restore
            
        Returns:
            Restored DataFrame
        """
        try:
            # Get backup metadata
            metadata = self._get_backup_metadata(backup_id)
            if not metadata:
                raise ValueError(f"Backup {backup_id} not found")
            
            # Find backup file
            backup_file = None
            for ext in [f".{metadata.format}", f".{metadata.format}.gz"]:
                potential_file = self.backup_dir / f"{backup_id}{ext}"
                if potential_file.exists():
                    backup_file = potential_file
                    break
            
            if not backup_file:
                raise FileNotFoundError(f"Backup file for {backup_id} not found")
            
            # Restore data
            if metadata.format == "parquet":
                data = pd.read_parquet(backup_file)
            elif metadata.format == "csv":
                data = pd.read_csv(backup_file)
            elif metadata.format == "json":
                data = pd.read_json(backup_file, orient='records')
            else:
                raise ValueError(f"Unsupported format: {metadata.format}")
            
            # Verify data integrity
            restored_hash = hashlib.sha256(data.to_string().encode()).hexdigest()
            if restored_hash != metadata.data_hash:
                logger.warning(f"Data hash mismatch for backup {backup_id}")
            
            logger.info(f"Restored backup {backup_id} with {len(data)} records")
            return data
            
        except Exception as e:
            logger.error(f"Failed to restore backup {backup_id}: {e}")
            raise
    
    def create_restore_point(
        self,
        backup_id: str,
        operation_type: str,
        affected_records: int,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create a restore point for rollback.
        
        Args:
            backup_id: Associated backup ID
            operation_type: Type of operation (cleaning, transformation, etc.)
            affected_records: Number of records affected
            metadata: Optional metadata
            
        Returns:
            Restore point ID
        """
        try:
            timestamp = time.time()
            restore_point_id = f"rp_{int(timestamp)}_{operation_type}"
            
            restore_point = RestorePoint(
                restore_point_id=restore_point_id,
                timestamp=timestamp,
                backup_id=backup_id,
                operation_type=operation_type,
                affected_records=affected_records,
                metadata=metadata or {}
            )
            
            # Store restore point
            self._store_restore_point(restore_point)
            
            logger.info(f"Created restore point {restore_point_id}")
            return restore_point_id
            
        except Exception as e:
            logger.error(f"Failed to create restore point: {e}")
            raise
    
    def list_backups(self) -> List[BackupMetadata]:
        """List all available backups."""
        try:
            with duckdb.connect(str(self.metadata_db)) as conn:
                result = conn.execute("""
                    SELECT * FROM backups 
                    ORDER BY timestamp DESC
                """).fetchall()
                
                backups = []
                for row in result:
                    backup = BackupMetadata(
                        backup_id=row[0],
                        timestamp=row[1],
                        source_info=json.loads(row[2]),
                        data_hash=row[3],
                        size_bytes=row[4],
                        format=row[5],
                        compression=row[6],
                        description=row[7]
                    )
                    backups.append(backup)
                
                return backups
                
        except Exception as e:
            logger.error(f"Failed to list backups: {e}")
            return []
    
    def list_restore_points(self, backup_id: Optional[str] = None) -> List[RestorePoint]:
        """List restore points, optionally filtered by backup ID."""
        try:
            with duckdb.connect(str(self.metadata_db)) as conn:
                if backup_id:
                    result = conn.execute("""
                        SELECT * FROM restore_points 
                        WHERE backup_id = ?
                        ORDER BY timestamp DESC
                    """, [backup_id]).fetchall()
                else:
                    result = conn.execute("""
                        SELECT * FROM restore_points 
                        ORDER BY timestamp DESC
                    """).fetchall()
                
                restore_points = []
                for row in result:
                    rp = RestorePoint(
                        restore_point_id=row[0],
                        timestamp=row[1],
                        backup_id=row[2],
                        operation_type=row[3],
                        affected_records=row[4],
                        metadata=json.loads(row[5])
                    )
                    restore_points.append(rp)
                
                return restore_points
                
        except Exception as e:
            logger.error(f"Failed to list restore points: {e}")
            return []
    
    def delete_backup(self, backup_id: str) -> bool:
        """Delete a backup and its associated files.
        
        Args:
            backup_id: ID of backup to delete
            
        Returns:
            True if deleted successfully
        """
        try:
            # Get backup metadata
            metadata = self._get_backup_metadata(backup_id)
            if not metadata:
                return False
            
            # Delete backup file
            backup_file = None
            for ext in [f".{metadata.format}", f".{metadata.format}.gz"]:
                potential_file = self.backup_dir / f"{backup_id}{ext}"
                if potential_file.exists():
                    potential_file.unlink()
                    backup_file = potential_file
                    break
            
            # Delete metadata
            with duckdb.connect(str(self.metadata_db)) as conn:
                # Delete restore points first (foreign key constraint)
                conn.execute("DELETE FROM restore_points WHERE backup_id = ?", [backup_id])
                
                # Delete backup metadata
                conn.execute("DELETE FROM backups WHERE backup_id = ?", [backup_id])
                conn.commit()
            
            logger.info(f"Deleted backup {backup_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete backup {backup_id}: {e}")
            return False
    
    def get_backup_stats(self) -> Dict[str, Any]:
        """Get backup system statistics."""
        try:
            backups = self.list_backups()
            restore_points = self.list_restore_points()
            
            if not backups:
                return {
                    "total_backups": 0,
                    "total_size_mb": 0,
                    "oldest_backup": None,
                    "newest_backup": None,
                    "total_restore_points": 0
                }
            
            total_size = sum(b.size_bytes for b in backups)
            oldest = min(backups, key=lambda b: b.timestamp)
            newest = max(backups, key=lambda b: b.timestamp)
            
            return {
                "total_backups": len(backups),
                "total_size_mb": total_size / (1024 * 1024),
                "oldest_backup": {
                    "id": oldest.backup_id,
                    "timestamp": oldest.timestamp,
                    "age_hours": (time.time() - oldest.timestamp) / 3600
                },
                "newest_backup": {
                    "id": newest.backup_id,
                    "timestamp": newest.timestamp,
                    "age_hours": (time.time() - newest.timestamp) / 3600
                },
                "total_restore_points": len(restore_points),
                "backup_directory": str(self.backup_dir),
                "metadata_db_size_mb": self.metadata_db.stat().st_size / (1024 * 1024)
            }
            
        except Exception as e:
            logger.error(f"Failed to get backup stats: {e}")
            return {}
    
    def _store_backup_metadata(self, metadata: BackupMetadata):
        """Store backup metadata in database."""
        with duckdb.connect(str(self.metadata_db)) as conn:
            conn.execute("""
                INSERT INTO backups 
                (backup_id, timestamp, source_info, data_hash, size_bytes, format, compression, description)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                metadata.backup_id,
                metadata.timestamp,
                json.dumps(metadata.source_info),
                metadata.data_hash,
                metadata.size_bytes,
                metadata.format,
                metadata.compression,
                metadata.description
            ])
            conn.commit()
    
    def _store_restore_point(self, restore_point: RestorePoint):
        """Store restore point in database."""
        with duckdb.connect(str(self.metadata_db)) as conn:
            conn.execute("""
                INSERT INTO restore_points 
                (restore_point_id, timestamp, backup_id, operation_type, affected_records, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
            """, [
                restore_point.restore_point_id,
                restore_point.timestamp,
                restore_point.backup_id,
                restore_point.operation_type,
                restore_point.affected_records,
                json.dumps(restore_point.metadata)
            ])
            conn.commit()
    
    def _get_backup_metadata(self, backup_id: str) -> Optional[BackupMetadata]:
        """Get backup metadata by ID."""
        try:
            with duckdb.connect(str(self.metadata_db)) as conn:
                result = conn.execute("""
                    SELECT * FROM backups WHERE backup_id = ?
                """, [backup_id]).fetchone()
                
                if not result:
                    return None
                
                return BackupMetadata(
                    backup_id=result[0],
                    timestamp=result[1],
                    source_info=json.loads(result[2]),
                    data_hash=result[3],
                    size_bytes=result[4],
                    format=result[5],
                    compression=result[6],
                    description=result[7]
                )
                
        except Exception as e:
            logger.error(f"Failed to get backup metadata: {e}")
            return None
    
    def _cleanup_old_backups(self):
        """Clean up old backups beyond the retention limit."""
        try:
            backups = self.list_backups()
            if len(backups) <= self.max_backups:
                return
            
            # Delete oldest backups
            backups_to_delete = backups[self.max_backups:]
            for backup in backups_to_delete:
                self.delete_backup(backup.backup_id)
            
            logger.info(f"Cleaned up {len(backups_to_delete)} old backups")
            
        except Exception as e:
            logger.error(f"Failed to cleanup old backups: {e}")


class AutoBackupWrapper:
    """Wrapper that automatically creates backups before operations."""
    
    def __init__(self, backup_manager: DataBackupManager):
        """Initialize auto-backup wrapper.
        
        Args:
            backup_manager: Backup manager instance
        """
        self.backup_manager = backup_manager
        self.enabled = True
    
    def backup_before_operation(self, data: pd.DataFrame, operation: str) -> Optional[str]:
        """Create backup before operation.
        
        Args:
            data: Data to backup
            operation: Description of operation
            
        Returns:
            Backup ID or None if disabled
        """
        if not self.enabled:
            return None
        
        try:
            backup_id = self.backup_manager.create_backup(
                data,
                description=f"Pre-{operation} backup"
            )
            
            # Create restore point
            self.backup_manager.create_restore_point(
                backup_id=backup_id,
                operation_type=operation,
                affected_records=len(data),
                metadata={"auto_backup": True}
            )
            
            return backup_id
            
        except Exception as e:
            logger.error(f"Auto-backup failed: {e}")
            return None
    
    def enable(self):
        """Enable auto-backup."""
        self.enabled = True
        logger.info("Auto-backup enabled")
    
    def disable(self):
        """Disable auto-backup."""
        self.enabled = False
        logger.info("Auto-backup disabled")


# Global backup manager
_global_backup_manager = None


def get_global_backup_manager(backup_dir: str = "./backups") -> DataBackupManager:
    """Get or create global backup manager."""
    global _global_backup_manager
    if _global_backup_manager is None:
        _global_backup_manager = DataBackupManager(backup_dir)
    return _global_backup_manager