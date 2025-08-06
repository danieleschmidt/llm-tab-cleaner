"""Incremental cleaning with state management."""

import hashlib
import json
import logging
import sqlite3
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Any

import pandas as pd

from .core import TableCleaner, CleaningReport, Fix


logger = logging.getLogger(__name__)


class IncrementalCleaner:
    """Handles incremental data cleaning with state persistence."""
    
    def __init__(
        self,
        state_path: str,
        llm_provider: str = "local",
        **kwargs
    ):
        """Initialize incremental cleaner.
        
        Args:
            state_path: Path to state database
            llm_provider: LLM provider to use
            **kwargs: Additional arguments for TableCleaner
        """
        self.state_path = Path(state_path)
        self.cleaner = TableCleaner(llm_provider=llm_provider, **kwargs)
        self._state_loaded = False
        self._processed_hashes: Set[str] = set()
        self._column_stats: Dict[str, Any] = {}
        self._cleaning_history: List[Dict[str, Any]] = []
        
        # Initialize state database
        self._init_state_db()
        
    def process_increment(
        self,
        new_records: pd.DataFrame,
        update_statistics: bool = True
    ) -> pd.DataFrame:
        """Process new data incrementally.
        
        Args:
            new_records: New records to process
            update_statistics: Whether to update cleaning statistics
            
        Returns:
            Cleaned DataFrame
        """
        logger.info(f"Processing incremental batch of {len(new_records)} records")
        
        # Load state if not already loaded
        if not self._state_loaded:
            self._load_state()
        
        # Filter out already processed records
        new_records_filtered = self._filter_processed_records(new_records)
        
        if len(new_records_filtered) == 0:
            logger.info("All records already processed, skipping")
            return new_records
        
        logger.info(f"Processing {len(new_records_filtered)} new records")
        
        # Clean new records
        cleaned_df, report = self.cleaner.clean(new_records_filtered)
        
        # Update state if requested
        if update_statistics:
            self._update_state(cleaned_df, report)
        
        # Store processed record hashes
        self._store_processed_hashes(cleaned_df)
        
        return cleaned_df
        
    def reprocess_low_confidence(
        self,
        confidence_threshold: float = 0.7,
        new_model: Optional[str] = None
    ) -> None:
        """Reprocess records with low confidence scores.
        
        Args:
            confidence_threshold: Threshold below which to reprocess
            new_model: Optional new model to use for reprocessing
        """
        logger.info(f"Reprocessing records with confidence < {confidence_threshold}")
        
        if new_model:
            logger.info(f"Using new model: {new_model}")
            # Update cleaner with new model
            self.cleaner = TableCleaner(
                llm_provider=new_model,
                confidence_threshold=self.cleaner.confidence_threshold,
                rules=self.cleaner.rules,
                enable_profiling=self.cleaner.enable_profiling,
                max_fixes_per_column=self.cleaner.max_fixes_per_column
            )
        
        # Query low confidence records from database
        low_confidence_records = self._get_low_confidence_records(confidence_threshold)
        
        if len(low_confidence_records) > 0:
            logger.info(f"Reprocessing {len(low_confidence_records)} low confidence records")
            # Remove from processed hashes to allow reprocessing
            for record_hash in low_confidence_records['record_hash']:
                self._processed_hashes.discard(record_hash)
            
            # Update database to mark for reprocessing
            self._mark_for_reprocessing(low_confidence_records['record_hash'].tolist())
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get cleaning statistics from state."""
        if not self._state_loaded:
            self._load_state()
        
        return {
            "total_processed_records": len(self._processed_hashes),
            "column_statistics": self._column_stats,
            "cleaning_history": self._cleaning_history[-10:]  # Last 10 runs
        }
    
    def _init_state_db(self) -> None:
        """Initialize SQLite state database."""
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        
        conn = sqlite3.connect(str(self.state_path))
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS processed_records (
                record_hash TEXT PRIMARY KEY,
                processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                confidence REAL,
                fixes_applied INTEGER DEFAULT 0
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS column_stats (
                column_name TEXT PRIMARY KEY,
                total_processed INTEGER DEFAULT 0,
                total_fixes INTEGER DEFAULT 0,
                avg_confidence REAL DEFAULT 1.0,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS cleaning_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                records_processed INTEGER,
                fixes_applied INTEGER,
                quality_score REAL,
                processing_time REAL
            )
        ''')
        
        conn.commit()
        conn.close()
        
        logger.info(f"Initialized state database at {self.state_path}")
        
    def _load_state(self) -> None:
        """Load cleaning state from database."""
        conn = sqlite3.connect(str(self.state_path))
        
        # Load processed record hashes
        cursor = conn.cursor()
        cursor.execute("SELECT record_hash FROM processed_records")
        self._processed_hashes = {row[0] for row in cursor.fetchall()}
        
        # Load column statistics
        cursor.execute("SELECT column_name, total_processed, total_fixes, avg_confidence FROM column_stats")
        for row in cursor.fetchall():
            self._column_stats[row[0]] = {
                "total_processed": row[1],
                "total_fixes": row[2],
                "avg_confidence": row[3]
            }
        
        # Load cleaning history
        cursor.execute("SELECT records_processed, fixes_applied, quality_score, processing_time FROM cleaning_history ORDER BY run_timestamp DESC LIMIT 50")
        self._cleaning_history = [
            {
                "records_processed": row[0],
                "fixes_applied": row[1],
                "quality_score": row[2],
                "processing_time": row[3]
            }
            for row in cursor.fetchall()
        ]
        
        conn.close()
        self._state_loaded = True
        
        logger.info(f"Loaded state: {len(self._processed_hashes)} processed records, {len(self._column_stats)} columns tracked")
        
    def _update_state(self, cleaned_df: pd.DataFrame, report: CleaningReport) -> None:
        """Update state with new cleaning results."""
        conn = sqlite3.connect(str(self.state_path))
        cursor = conn.cursor()
        
        # Update column statistics
        for column in cleaned_df.columns:
            column_fixes = len([f for f in report.fixes if f.column == column])
            avg_confidence = sum([f.confidence for f in report.fixes if f.column == column]) / max(1, column_fixes)
            
            if column in self._column_stats:
                # Update existing stats
                cursor.execute('''
                    UPDATE column_stats 
                    SET total_processed = total_processed + ?,
                        total_fixes = total_fixes + ?,
                        avg_confidence = (avg_confidence * total_processed + ? * ?) / (total_processed + ?),
                        last_updated = CURRENT_TIMESTAMP
                    WHERE column_name = ?
                ''', (len(cleaned_df), column_fixes, avg_confidence, column_fixes, len(cleaned_df), column))
            else:
                # Insert new stats
                cursor.execute('''
                    INSERT INTO column_stats (column_name, total_processed, total_fixes, avg_confidence)
                    VALUES (?, ?, ?, ?)
                ''', (column, len(cleaned_df), column_fixes, avg_confidence))
        
        # Add cleaning history entry
        cursor.execute('''
            INSERT INTO cleaning_history (records_processed, fixes_applied, quality_score, processing_time)
            VALUES (?, ?, ?, ?)
        ''', (len(cleaned_df), report.total_fixes, report.quality_score, report.processing_time))
        
        conn.commit()
        conn.close()
        
    def _filter_processed_records(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter out already processed records."""
        if len(self._processed_hashes) == 0:
            return df
        
        # Calculate hashes for all records
        record_hashes = df.apply(self._calculate_record_hash, axis=1)
        
        # Filter unprocessed records
        unprocessed_mask = ~record_hashes.isin(self._processed_hashes)
        return df[unprocessed_mask]
    
    def _store_processed_hashes(self, df: pd.DataFrame) -> None:
        """Store record hashes in database."""
        conn = sqlite3.connect(str(self.state_path))
        cursor = conn.cursor()
        
        for _, row in df.iterrows():
            record_hash = self._calculate_record_hash(row)
            self._processed_hashes.add(record_hash)
            
            cursor.execute('''
                INSERT OR IGNORE INTO processed_records (record_hash, confidence)
                VALUES (?, ?)
            ''', (record_hash, 1.0))  # Default confidence
        
        conn.commit()
        conn.close()
    
    def _calculate_record_hash(self, row: pd.Series) -> str:
        """Calculate SHA-256 hash of a record."""
        # Convert row to string and hash
        row_str = '|'.join([str(val) for val in row.values])
        return hashlib.sha256(row_str.encode()).hexdigest()
    
    def _get_low_confidence_records(self, threshold: float) -> pd.DataFrame:
        """Get records with confidence below threshold."""
        conn = sqlite3.connect(str(self.state_path))
        
        query = '''
            SELECT record_hash, confidence 
            FROM processed_records 
            WHERE confidence < ?
            ORDER BY confidence ASC
        '''
        
        df = pd.read_sql_query(query, conn, params=(threshold,))
        conn.close()
        
        return df
    
    def _mark_for_reprocessing(self, record_hashes: List[str]) -> None:
        """Mark records for reprocessing by removing from processed set."""
        conn = sqlite3.connect(str(self.state_path))
        cursor = conn.cursor()
        
        # Delete from processed records
        placeholders = ','.join(['?' for _ in record_hashes])
        cursor.execute(f'DELETE FROM processed_records WHERE record_hash IN ({placeholders})', record_hashes)
        
        conn.commit()
        conn.close()
        
        logger.info(f"Marked {len(record_hashes)} records for reprocessing")