#!/usr/bin/env python3
"""
Model Cleanup Utility for UK Road Risk Classification System

This script manages model storage by:
1. Keeping only the N most recent model versions
2. Cleaning up orphaned model files
3. Maintaining model performance history
4. Providing storage usage statistics

Usage:
    python scripts/cleanup_models.py --keep 5
    python scripts/cleanup_models.py --dry-run
    python scripts/cleanup_models.py --stats
"""

import os
import shutil
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelCleanup:
    """Utility for cleaning up old ML model versions while preserving the best performers"""
    
    def __init__(self, models_dir: str = "data/models", keep_versions: int = 5):
        self.models_dir = Path(models_dir)
        self.keep_versions = keep_versions
        self.cleanup_report = {
            'timestamp': datetime.now().isoformat(),
            'deleted_models': [],
            'kept_models': [],
            'space_freed': 0,
            'total_space': 0
        }
    
    def get_model_versions(self) -> List[Tuple[str, datetime, Path]]:
        """Get all model versions sorted by creation date"""
        model_versions = []
        
        for item in self.models_dir.iterdir():
            if item.is_dir() and item.name.startswith('model_v_'):
                try:
                    # Extract datetime from folder name: model_v_YYYYMMDD_HHMMSS
                    date_str = item.name.replace('model_v_', '')
                    model_date = datetime.strptime(date_str, '%Y%m%d_%H%M%S')
                    model_versions.append((item.name, model_date, item))
                except ValueError:
                    logger.warning(f"Skipping invalid model folder: {item.name}")
                    continue
        
        # Sort by date (newest first)
        model_versions.sort(key=lambda x: x[1], reverse=True)
        return model_versions
    
    def get_model_performance(self, model_path: Path) -> Dict:
        """Extract model performance from metadata.json"""
        metadata_file = model_path / 'metadata.json'
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                    return metadata.get('performance', {})
            except Exception as e:
                logger.warning(f"Failed to read metadata for {model_path.name}: {e}")
        return {}
    
    def get_directory_size(self, path: Path) -> int:
        """Calculate directory size in bytes"""
        total_size = 0
        try:
            for dirpath, dirnames, filenames in os.walk(path):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    try:
                        total_size += os.path.getsize(filepath)
                    except OSError:
                        pass
        except Exception:
            pass
        return total_size
    
    def format_size(self, size_bytes: int) -> str:
        """Format size in human-readable format"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024
        return f"{size_bytes:.1f} TB"
    
    def cleanup_old_models(self, dry_run: bool = False) -> Dict:
        """Remove old model versions, keeping the most recent ones"""
        model_versions = self.get_model_versions()
        
        if len(model_versions) <= self.keep_versions:
            logger.info(f"Only {len(model_versions)} model versions found. Nothing to clean up.")
            return self.cleanup_report
        
        models_to_delete = model_versions[self.keep_versions:]
        models_to_keep = model_versions[:self.keep_versions]
        
        # Calculate total space usage
        total_space = sum(self.get_directory_size(path) for _, _, path in model_versions)
        self.cleanup_report['total_space'] = total_space
        
        logger.info(f"Found {len(model_versions)} model versions")
        logger.info(f"Keeping {len(models_to_keep)} most recent versions")
        logger.info(f"Deleting {len(models_to_delete)} old versions")
        
        # Log models being kept
        for name, date, path in models_to_keep:
            performance = self.get_model_performance(path)
            size = self.get_directory_size(path)
            accuracy = performance.get('accuracy', 'N/A')
            
            self.cleanup_report['kept_models'].append({
                'name': name,
                'date': date.isoformat(),
                'size': size,
                'accuracy': accuracy
            })
            
            logger.info(f"✅ KEEPING: {name} (Accuracy: {accuracy}, Size: {self.format_size(size)})")
        
        # Delete old models
        space_freed = 0
        for name, date, path in models_to_delete:
            performance = self.get_model_performance(path)
            size = self.get_directory_size(path)
            accuracy = performance.get('accuracy', 'N/A')
            
            if dry_run:
                logger.info(f"🗑️  WOULD DELETE: {name} (Accuracy: {accuracy}, Size: {self.format_size(size)})")
            else:
                try:
                    shutil.rmtree(path)
                    space_freed += size
                    logger.info(f"🗑️  DELETED: {name} (Accuracy: {accuracy}, Size: {self.format_size(size)})")
                    
                    self.cleanup_report['deleted_models'].append({
                        'name': name,
                        'date': date.isoformat(),
                        'size': size,
                        'accuracy': accuracy
                    })
                except Exception as e:
                    logger.error(f"Failed to delete {name}: {e}")
        
        self.cleanup_report['space_freed'] = space_freed
        
        if not dry_run and space_freed > 0:
            logger.info(f"🎉 Cleanup completed! Freed {self.format_size(space_freed)} of storage")
        
        return self.cleanup_report
    
    def cleanup_orphaned_files(self, dry_run: bool = False) -> List[str]:
        """Remove orphaned model files (files without corresponding directories)"""
        orphaned_files = []
        
        for item in self.models_dir.iterdir():
            if item.is_file() and not item.name.startswith('.'):
                # Check if it's a standalone model file without a version directory
                if item.suffix in ['.pkl', '.json'] and not item.name.startswith('training_'):
                    orphaned_files.append(item.name)
                    
                    if dry_run:
                        logger.info(f"🗑️  WOULD DELETE ORPHANED: {item.name}")
                    else:
                        try:
                            item.unlink()
                            logger.info(f"🗑️  DELETED ORPHANED: {item.name}")
                        except Exception as e:
                            logger.error(f"Failed to delete orphaned file {item.name}: {e}")
        
        return orphaned_files
    
    def show_statistics(self) -> None:
        """Display model storage statistics"""
        model_versions = self.get_model_versions()
        
        if not model_versions:
            logger.info("No model versions found.")
            return
        
        total_size = sum(self.get_directory_size(path) for _, _, path in model_versions)
        
        print("\n" + "="*80)
        print("🗂️  MODEL STORAGE STATISTICS")
        print("="*80)
        print(f"📁 Models directory: {self.models_dir}")
        print(f"📊 Total model versions: {len(model_versions)}")
        print(f"💾 Total storage used: {self.format_size(total_size)}")
        print(f"📈 Average model size: {self.format_size(total_size / len(model_versions))}")
        print()
        
        print("📋 MODEL VERSION DETAILS:")
        print("-" * 80)
        print(f"{'Model Version':<25} {'Date':<20} {'Accuracy':<10} {'Size':<10}")
        print("-" * 80)
        
        for name, date, path in model_versions:
            performance = self.get_model_performance(path)
            size = self.get_directory_size(path)
            accuracy = performance.get('accuracy', 'N/A')
            
            if isinstance(accuracy, float):
                accuracy = f"{accuracy:.2%}"
            
            print(f"{name:<25} {date.strftime('%Y-%m-%d %H:%M'):<20} {str(accuracy):<10} {self.format_size(size):<10}")
        
        print("="*80)
    
    def save_cleanup_report(self, report_path: str = None) -> None:
        """Save cleanup report to JSON file"""
        if report_path is None:
            report_path = self.models_dir / f"cleanup_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(report_path, 'w') as f:
            json.dump(self.cleanup_report, f, indent=2)
        
        logger.info(f"Cleanup report saved to: {report_path}")

def main():
    parser = argparse.ArgumentParser(description="ML Model Cleanup Utility")
    parser.add_argument('--keep', type=int, default=5, 
                       help='Number of recent model versions to keep (default: 5)')
    parser.add_argument('--models-dir', default='data/models',
                       help='Path to models directory (default: data/models)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be deleted without actually deleting')
    parser.add_argument('--stats', action='store_true',
                       help='Show storage statistics only')
    parser.add_argument('--cleanup-orphans', action='store_true',
                       help='Also clean up orphaned files')
    parser.add_argument('--save-report', action='store_true',
                       help='Save cleanup report to JSON file')
    
    args = parser.parse_args()
    
    # Initialize cleanup utility
    cleanup = ModelCleanup(models_dir=args.models_dir, keep_versions=args.keep)
    
    if args.stats:
        cleanup.show_statistics()
        return
    
    # Perform cleanup
    logger.info(f"Starting model cleanup (keeping {args.keep} versions)")
    if args.dry_run:
        logger.info("🔍 DRY RUN MODE - No files will be deleted")
    
    # Clean up old model versions
    report = cleanup.cleanup_old_models(dry_run=args.dry_run)
    
    # Clean up orphaned files if requested
    if args.cleanup_orphans:
        orphaned = cleanup.cleanup_orphaned_files(dry_run=args.dry_run)
        if orphaned:
            logger.info(f"Found {len(orphaned)} orphaned files")
    
    # Save cleanup report if requested
    if args.save_report:
        cleanup.save_cleanup_report()
    
    # Summary
    if not args.dry_run:
        deleted_count = len(report['deleted_models'])
        space_freed = cleanup.format_size(report['space_freed'])
        logger.info(f"✅ Cleanup completed: {deleted_count} models deleted, {space_freed} freed")

if __name__ == '__main__':
    main()