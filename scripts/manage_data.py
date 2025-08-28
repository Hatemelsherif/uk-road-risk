#!/usr/bin/env python3
"""
Data Management Utility for UK Road Risk Classification System

This script provides comprehensive data management including:
1. Data persistence verification
2. Volume health checks  
3. Data backup and restore
4. Storage usage monitoring
5. Container data synchronization

Usage:
    python scripts/manage_data.py --check-volumes
    python scripts/manage_data.py --backup
    python scripts/manage_data.py --sync-to-container
    python scripts/manage_data.py --storage-report
"""

import os
import shutil
import json
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataManager:
    """Comprehensive data management for containerized ML application"""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.data_dir = self.project_root / "data"
        self.docker_compose_file = self.project_root / "docker-compose.yml"
        
        # Data directories
        self.directories = {
            'raw': self.data_dir / 'raw',
            'processed': self.data_dir / 'processed', 
            'models': self.data_dir / 'models'
        }
    
    def check_data_persistence(self) -> Dict:
        """Verify data persistence setup and volume mounts"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'directories': {},
            'docker_volumes': {},
            'recommendations': []
        }
        
        logger.info("🔍 Checking data persistence setup...")
        
        # Check local directories
        for name, path in self.directories.items():
            exists = path.exists()
            is_dir = path.is_dir() if exists else False
            size = self._get_directory_size(path) if exists and is_dir else 0
            file_count = len(list(path.rglob('*'))) if exists and is_dir else 0
            
            report['directories'][name] = {
                'path': str(path),
                'exists': exists,
                'is_directory': is_dir,
                'size_bytes': size,
                'size_formatted': self._format_size(size),
                'file_count': file_count
            }
            
            status = "✅" if exists and is_dir else "❌"
            logger.info(f"{status} {name}: {path} ({self._format_size(size)}, {file_count} files)")
            
            if not exists:
                report['recommendations'].append(f"Create missing directory: {path}")
            elif not is_dir:
                report['recommendations'].append(f"Path exists but is not a directory: {path}")
        
        # Check Docker volumes
        if self._is_docker_available():
            volumes = self._get_docker_volumes()
            report['docker_volumes'] = volumes
            
            for vol_name, vol_info in volumes.items():
                logger.info(f"🐳 Docker volume '{vol_name}': {vol_info['size_formatted']}")
        
        return report
    
    def create_backup(self, backup_dir: str = None) -> str:
        """Create backup of all data directories"""
        if backup_dir is None:
            backup_dir = self.project_root / "backups" / f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        backup_path = Path(backup_dir)
        backup_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"📦 Creating backup at: {backup_path}")
        
        # Backup each data directory
        for name, source_path in self.directories.items():
            if source_path.exists():
                dest_path = backup_path / name
                logger.info(f"Backing up {name}: {source_path} -> {dest_path}")
                
                try:
                    shutil.copytree(source_path, dest_path, dirs_exist_ok=True)
                    logger.info(f"✅ {name} backup completed")
                except Exception as e:
                    logger.error(f"❌ Failed to backup {name}: {e}")
        
        # Create backup manifest
        manifest = {
            'created': datetime.now().isoformat(),
            'source_directories': {name: str(path) for name, path in self.directories.items()},
            'backup_size': self._get_directory_size(backup_path),
            'file_count': len(list(backup_path.rglob('*')))
        }
        
        manifest_file = backup_path / 'backup_manifest.json'
        with open(manifest_file, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        total_size = self._format_size(manifest['backup_size'])
        logger.info(f"🎉 Backup completed: {backup_path} ({total_size})")
        
        return str(backup_path)
    
    def sync_to_container(self, container_name: str = "uk-road-risk-web-1") -> bool:
        """Sync local data to running container"""
        if not self._is_docker_available():
            logger.error("Docker is not available")
            return False
        
        # Check if container is running
        if not self._is_container_running(container_name):
            logger.error(f"Container {container_name} is not running")
            return False
        
        logger.info(f"🔄 Syncing data to container: {container_name}")
        
        # Sync each directory
        for name, local_path in self.directories.items():
            if local_path.exists():
                container_path = f"/app/data/{name}"
                logger.info(f"Syncing {name}: {local_path} -> {container_name}:{container_path}")
                
                try:
                    # Use docker cp command
                    cmd = f"docker cp {local_path}/. {container_name}:{container_path}/"
                    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                    
                    if result.returncode == 0:
                        logger.info(f"✅ {name} sync completed")
                    else:
                        logger.error(f"❌ Failed to sync {name}: {result.stderr}")
                        return False
                        
                except Exception as e:
                    logger.error(f"❌ Failed to sync {name}: {e}")
                    return False
        
        logger.info("🎉 Data synchronization completed")
        return True
    
    def generate_storage_report(self) -> Dict:
        """Generate comprehensive storage usage report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'local_storage': {},
            'docker_storage': {},
            'recommendations': [],
            'cleanup_suggestions': []
        }
        
        logger.info("📊 Generating storage report...")
        
        # Analyze local storage
        total_local_size = 0
        for name, path in self.directories.items():
            if path.exists():
                size = self._get_directory_size(path)
                file_count = len(list(path.rglob('*')))
                
                # Analyze subdirectories
                subdirs = {}
                if path.is_dir():
                    for subdir in path.iterdir():
                        if subdir.is_dir():
                            sub_size = self._get_directory_size(subdir)
                            sub_count = len(list(subdir.rglob('*')))
                            subdirs[subdir.name] = {
                                'size_bytes': sub_size,
                                'size_formatted': self._format_size(sub_size),
                                'file_count': sub_count
                            }
                
                report['local_storage'][name] = {
                    'size_bytes': size,
                    'size_formatted': self._format_size(size),
                    'file_count': file_count,
                    'subdirectories': subdirs
                }
                
                total_local_size += size
                
                # Generate cleanup suggestions for models directory
                if name == 'models' and path.exists():
                    model_versions = list(path.glob('model_v_*'))
                    if len(model_versions) > 5:
                        old_models_size = sum(self._get_directory_size(mv) for mv in model_versions[5:])
                        report['cleanup_suggestions'].append({
                            'type': 'old_models',
                            'description': f'Delete {len(model_versions) - 5} old model versions',
                            'potential_savings': self._format_size(old_models_size)
                        })
        
        report['local_storage']['total'] = {
            'size_bytes': total_local_size,
            'size_formatted': self._format_size(total_local_size)
        }
        
        # Analyze Docker storage
        if self._is_docker_available():
            report['docker_storage'] = self._get_docker_storage_info()
        
        # Generate recommendations
        if total_local_size > 1024 * 1024 * 1024:  # > 1GB
            report['recommendations'].append("Consider implementing automated cleanup for large datasets")
        
        if len(report['cleanup_suggestions']) > 0:
            report['recommendations'].append("Run model cleanup to free storage space")
        
        return report
    
    def cleanup_temp_files(self) -> List[str]:
        """Clean up temporary and cache files"""
        temp_patterns = [
            '**/__pycache__',
            '**/*.pyc',
            '**/.DS_Store',
            '**/Thumbs.db',
            '**/*.tmp',
            '**/.*~'
        ]
        
        cleaned_files = []
        
        logger.info("🧹 Cleaning temporary files...")
        
        for pattern in temp_patterns:
            for temp_file in self.project_root.rglob(pattern):
                try:
                    if temp_file.is_file():
                        temp_file.unlink()
                        cleaned_files.append(str(temp_file))
                        logger.info(f"Deleted: {temp_file}")
                    elif temp_file.is_dir():
                        shutil.rmtree(temp_file)
                        cleaned_files.append(str(temp_file))
                        logger.info(f"Deleted directory: {temp_file}")
                except Exception as e:
                    logger.warning(f"Failed to delete {temp_file}: {e}")
        
        logger.info(f"🎉 Cleaned {len(cleaned_files)} temporary files")
        return cleaned_files
    
    def _get_directory_size(self, path: Path) -> int:
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
    
    def _format_size(self, size_bytes: int) -> str:
        """Format size in human-readable format"""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024
        return f"{size_bytes:.1f} PB"
    
    def _is_docker_available(self) -> bool:
        """Check if Docker is available"""
        try:
            subprocess.run(['docker', '--version'], capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False
    
    def _is_container_running(self, container_name: str) -> bool:
        """Check if container is running"""
        try:
            result = subprocess.run(
                ['docker', 'ps', '--format', '{{.Names}}'], 
                capture_output=True, text=True, check=True
            )
            return container_name in result.stdout
        except subprocess.CalledProcessError:
            return False
    
    def _get_docker_volumes(self) -> Dict:
        """Get information about Docker volumes"""
        volumes = {}
        try:
            # List volumes
            result = subprocess.run(
                ['docker', 'volume', 'ls', '--format', '{{.Name}}'],
                capture_output=True, text=True, check=True
            )
            
            for vol_name in result.stdout.strip().split('\n'):
                if vol_name and 'uk-road-risk' in vol_name:
                    # Get volume details
                    vol_result = subprocess.run(
                        ['docker', 'volume', 'inspect', vol_name],
                        capture_output=True, text=True, check=True
                    )
                    
                    vol_info = json.loads(vol_result.stdout)[0]
                    mount_point = vol_info['Mountpoint']
                    
                    # Estimate size (this is approximate)
                    try:
                        size_result = subprocess.run(
                            ['du', '-sb', mount_point],
                            capture_output=True, text=True
                        )
                        size_bytes = int(size_result.stdout.split()[0]) if size_result.returncode == 0 else 0
                    except:
                        size_bytes = 0
                    
                    volumes[vol_name] = {
                        'mountpoint': mount_point,
                        'size_bytes': size_bytes,
                        'size_formatted': self._format_size(size_bytes)
                    }
        
        except subprocess.CalledProcessError:
            logger.warning("Failed to get Docker volume information")
        
        return volumes
    
    def _get_docker_storage_info(self) -> Dict:
        """Get Docker storage information"""
        storage_info = {}
        
        try:
            # Get system info
            result = subprocess.run(
                ['docker', 'system', 'df', '--format', 'json'],
                capture_output=True, text=True, check=True
            )
            
            # Parse the output (note: docker system df --format json is not standard)
            # Fall back to regular parsing
            result = subprocess.run(
                ['docker', 'system', 'df'],
                capture_output=True, text=True, check=True
            )
            
            storage_info['raw_output'] = result.stdout
            
        except subprocess.CalledProcessError:
            logger.warning("Failed to get Docker storage information")
        
        return storage_info

def main():
    parser = argparse.ArgumentParser(description="Data Management Utility")
    parser.add_argument('--check-volumes', action='store_true',
                       help='Check data persistence and volume health')
    parser.add_argument('--backup', action='store_true',
                       help='Create backup of all data directories')
    parser.add_argument('--backup-dir', 
                       help='Custom backup directory path')
    parser.add_argument('--sync-to-container', action='store_true',
                       help='Sync local data to running container')
    parser.add_argument('--container-name', default='uk-road-risk-web-1',
                       help='Container name for sync operations')
    parser.add_argument('--storage-report', action='store_true',
                       help='Generate comprehensive storage usage report')
    parser.add_argument('--cleanup-temp', action='store_true',
                       help='Clean up temporary and cache files')
    parser.add_argument('--project-root', default='.',
                       help='Project root directory (default: current directory)')
    
    args = parser.parse_args()
    
    # Initialize data manager
    manager = DataManager(project_root=args.project_root)
    
    if args.check_volumes:
        logger.info("🔍 Checking data persistence setup...")
        report = manager.check_data_persistence()
        
        print("\n" + "="*80)
        print("📁 DATA PERSISTENCE REPORT")
        print("="*80)
        
        for name, info in report['directories'].items():
            status = "✅" if info['exists'] and info['is_directory'] else "❌"
            print(f"{status} {name.upper()}: {info['path']}")
            print(f"   Size: {info['size_formatted']}, Files: {info['file_count']}")
        
        if report['recommendations']:
            print("\n🔧 RECOMMENDATIONS:")
            for rec in report['recommendations']:
                print(f"   • {rec}")
        
        print("="*80)
    
    elif args.backup:
        backup_path = manager.create_backup(args.backup_dir)
        logger.info(f"✅ Backup created at: {backup_path}")
    
    elif args.sync_to_container:
        success = manager.sync_to_container(args.container_name)
        if success:
            logger.info("✅ Data synchronization completed successfully")
        else:
            logger.error("❌ Data synchronization failed")
    
    elif args.storage_report:
        report = manager.generate_storage_report()
        
        print("\n" + "="*80)
        print("📊 STORAGE USAGE REPORT")
        print("="*80)
        
        print("\n📁 LOCAL STORAGE:")
        for name, info in report['local_storage'].items():
            if name != 'total':
                print(f"   {name.upper()}: {info['size_formatted']} ({info['file_count']} files)")
                
                # Show subdirectories if any
                if 'subdirectories' in info and info['subdirectories']:
                    for subdir, subinfo in info['subdirectories'].items():
                        print(f"     └─ {subdir}: {subinfo['size_formatted']} ({subinfo['file_count']} files)")
        
        if 'total' in report['local_storage']:
            print(f"\n💾 TOTAL LOCAL: {report['local_storage']['total']['size_formatted']}")
        
        if report['cleanup_suggestions']:
            print("\n🧹 CLEANUP SUGGESTIONS:")
            for suggestion in report['cleanup_suggestions']:
                print(f"   • {suggestion['description']} (Save: {suggestion['potential_savings']})")
        
        if report['recommendations']:
            print("\n🔧 RECOMMENDATIONS:")
            for rec in report['recommendations']:
                print(f"   • {rec}")
        
        print("="*80)
    
    elif args.cleanup_temp:
        cleaned = manager.cleanup_temp_files()
        logger.info(f"✅ Cleaned {len(cleaned)} temporary files")
    
    else:
        parser.print_help()

if __name__ == '__main__':
    main()