#!/usr/bin/env python3
"""
Production Hardening Module
Memory management, network resilience, performance monitoring, and production safety features
"""

import gc
import os
import sys
import time
import psutil
import threading
import traceback
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
from contextlib import contextmanager
import streamlit as st
import logging

class MemoryManager:
    """Memory management and optimization"""
    
    def __init__(self):
        self.memory_threshold = 0.85  # 85% memory usage threshold
        self.cleanup_callbacks = []
        self.memory_history = []
        
        # Initialize session state for memory tracking
        if 'memory_state' not in st.session_state:
            st.session_state.memory_state = {
                'peak_usage': 0,
                'cleanup_count': 0,
                'warnings_issued': 0,
                'last_cleanup': None
            }
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics"""
        process = psutil.Process()
        memory_info = process.memory_info()
        system_memory = psutil.virtual_memory()
        
        return {
            'process_mb': memory_info.rss / 1024 / 1024,
            'process_percent': process.memory_percent(),
            'system_percent': system_memory.percent,
            'system_available_mb': system_memory.available / 1024 / 1024,
            'system_total_mb': system_memory.total / 1024 / 1024
        }
    
    def check_memory_pressure(self) -> bool:
        """Check if system is under memory pressure"""
        usage = self.get_memory_usage()
        return usage['system_percent'] > self.memory_threshold * 100
    
    def register_cleanup_callback(self, callback: Callable):
        """Register a cleanup callback for memory pressure"""
        self.cleanup_callbacks.append(callback)
    
    def cleanup_memory(self, force: bool = False):
        """Perform memory cleanup"""
        if not force and not self.check_memory_pressure():
            return False
        
        initial_usage = self.get_memory_usage()
        
        # Run cleanup callbacks
        for callback in self.cleanup_callbacks:
            try:
                callback()
            except Exception as e:
                logging.error(f"Memory cleanup callback failed: {e}")
        
        # Clear caches
        if hasattr(st, 'cache_data'):
            st.cache_data.clear()
        if hasattr(st, 'cache_resource'):
            st.cache_resource.clear()
        
        # Force garbage collection
        gc.collect()
        
        # Update session state
        st.session_state.memory_state['cleanup_count'] += 1
        st.session_state.memory_state['last_cleanup'] = datetime.now().isoformat()
        
        final_usage = self.get_memory_usage()
        freed_mb = initial_usage['process_mb'] - final_usage['process_mb']
        
        logging.info(f"Memory cleanup completed. Freed: {freed_mb:.1f}MB")
        return True
    
    @contextmanager
    def memory_monitor(self, operation_name: str):
        """Context manager for monitoring memory usage during operations"""
        start_usage = self.get_memory_usage()
        start_time = time.time()
        
        try:
            yield
        finally:
            end_usage = self.get_memory_usage()
            end_time = time.time()
            
            memory_delta = end_usage['process_mb'] - start_usage['process_mb']
            duration = end_time - start_time
            
            # Update peak usage
            if end_usage['process_mb'] > st.session_state.memory_state['peak_usage']:
                st.session_state.memory_state['peak_usage'] = end_usage['process_mb']
            
            # Log memory usage
            logging.info(f"Memory usage for {operation_name}: {memory_delta:+.1f}MB in {duration:.1f}s")
            
            # Check for memory pressure
            if self.check_memory_pressure():
                st.session_state.memory_state['warnings_issued'] += 1
                st.warning(f"⚠️ High memory usage detected: {end_usage['system_percent']:.1f}%")
                self.cleanup_memory()

class NetworkResilience:
    """Network resilience and timeout handling"""
    
    def __init__(self):
        self.default_timeout = 30
        self.max_retries = 3
        self.backoff_factor = 2
        self.connection_pool = {}
        
        # Initialize session state for network tracking
        if 'network_state' not in st.session_state:
            st.session_state.network_state = {
                'total_requests': 0,
                'failed_requests': 0,
                'timeout_count': 0,
                'retry_count': 0,
                'last_failure': None,
                'connection_issues': []
            }
    
    def configure_timeouts(self, operation_type: str) -> Dict[str, int]:
        """Configure timeouts based on operation type"""
        timeout_configs = {
            'api_call': {'connect': 10, 'read': 30, 'total': 60},
            'file_upload': {'connect': 15, 'read': 120, 'total': 300},
            'file_download': {'connect': 10, 'read': 60, 'total': 180},
            'processing': {'connect': 5, 'read': 300, 'total': 600},
            'validation': {'connect': 5, 'read': 30, 'total': 60}
        }
        
        return timeout_configs.get(operation_type, timeout_configs['api_call'])
    
    @contextmanager
    def network_operation(self, operation_name: str, timeout_config: Optional[Dict] = None):
        """Context manager for network operations with resilience"""
        if timeout_config is None:
            timeout_config = self.configure_timeouts('api_call')
        
        start_time = time.time()
        st.session_state.network_state['total_requests'] += 1
        
        try:
            yield timeout_config
            
        except Exception as e:
            # Track network failures
            st.session_state.network_state['failed_requests'] += 1
            st.session_state.network_state['last_failure'] = datetime.now().isoformat()
            
            error_type = type(e).__name__
            if 'timeout' in str(e).lower() or 'timed out' in str(e).lower():
                st.session_state.network_state['timeout_count'] += 1
            
            # Log network issue
            issue = {
                'timestamp': datetime.now().isoformat(),
                'operation': operation_name,
                'error_type': error_type,
                'error_message': str(e),
                'duration': time.time() - start_time
            }
            
            st.session_state.network_state['connection_issues'].append(issue)
            
            # Keep only recent issues (last 50)
            if len(st.session_state.network_state['connection_issues']) > 50:
                st.session_state.network_state['connection_issues'] = \
                    st.session_state.network_state['connection_issues'][-50:]
            
            logging.error(f"Network operation {operation_name} failed: {e}")
            raise
    
    def get_network_health(self) -> Dict[str, Any]:
        """Get network health statistics"""
        state = st.session_state.network_state
        
        total = state['total_requests']
        failed = state['failed_requests']
        
        success_rate = (total - failed) / total if total > 0 else 1.0
        
        return {
            'success_rate': success_rate,
            'total_requests': total,
            'failed_requests': failed,
            'timeout_count': state['timeout_count'],
            'retry_count': state['retry_count'],
            'last_failure': state['last_failure'],
            'health_status': 'good' if success_rate > 0.95 else 'degraded' if success_rate > 0.8 else 'poor'
        }

class PerformanceMonitor:
    """Performance monitoring and optimization"""
    
    def __init__(self):
        self.performance_history = []
        self.operation_benchmarks = {}
        
        # Initialize session state for performance tracking
        if 'performance_state' not in st.session_state:
            st.session_state.performance_state = {
                'operations': [],
                'slow_operations': [],
                'performance_alerts': [],
                'baseline_metrics': {},
                'optimization_suggestions': []
            }
    
    @contextmanager
    def performance_tracker(self, operation_name: str, expected_duration: Optional[float] = None):
        """Context manager for tracking operation performance"""
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        try:
            yield
            
        finally:
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            duration = end_time - start_time
            memory_delta = end_memory - start_memory
            
            # Record operation
            operation_record = {
                'name': operation_name,
                'duration': duration,
                'memory_delta': memory_delta,
                'timestamp': datetime.now().isoformat(),
                'expected_duration': expected_duration
            }
            
            st.session_state.performance_state['operations'].append(operation_record)
            
            # Keep only recent operations (last 100)
            if len(st.session_state.performance_state['operations']) > 100:
                st.session_state.performance_state['operations'] = \
                    st.session_state.performance_state['operations'][-100:]
            
            # Check for slow operations
            if expected_duration and duration > expected_duration * 2:
                slow_operation = {
                    'operation': operation_name,
                    'duration': duration,
                    'expected': expected_duration,
                    'slowdown_factor': duration / expected_duration,
                    'timestamp': datetime.now().isoformat()
                }
                
                st.session_state.performance_state['slow_operations'].append(slow_operation)
                
                # Generate performance alert
                alert = {
                    'type': 'slow_operation',
                    'message': f"{operation_name} took {duration:.1f}s (expected {expected_duration:.1f}s)",
                    'severity': 'high' if duration > expected_duration * 5 else 'medium',
                    'timestamp': datetime.now().isoformat(),
                    'data': slow_operation
                }
                
                st.session_state.performance_state['performance_alerts'].append(alert)
            
            # Update benchmarks
            if operation_name not in self.operation_benchmarks:
                self.operation_benchmarks[operation_name] = []
            
            self.operation_benchmarks[operation_name].append(duration)
            
            # Keep only recent benchmarks (last 20 per operation)
            if len(self.operation_benchmarks[operation_name]) > 20:
                self.operation_benchmarks[operation_name] = \
                    self.operation_benchmarks[operation_name][-20:]
            
            logging.info(f"Performance: {operation_name} completed in {duration:.2f}s (+{memory_delta:.1f}MB)")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary and statistics"""
        operations = st.session_state.performance_state['operations']
        
        if not operations:
            return {'status': 'no_data'}
        
        # Calculate statistics
        durations = [op['duration'] for op in operations]
        memory_deltas = [op['memory_delta'] for op in operations]
        
        avg_duration = sum(durations) / len(durations)
        avg_memory = sum(memory_deltas) / len(memory_deltas)
        
        # Recent performance (last 10 operations)
        recent_operations = operations[-10:]
        recent_durations = [op['duration'] for op in recent_operations]
        recent_avg = sum(recent_durations) / len(recent_durations) if recent_durations else 0
        
        # Performance trend
        trend = 'stable'
        if len(durations) >= 10:
            first_half_avg = sum(durations[:len(durations)//2]) / (len(durations)//2)
            second_half_avg = sum(durations[len(durations)//2:]) / (len(durations) - len(durations)//2)
            
            if second_half_avg > first_half_avg * 1.2:
                trend = 'degrading'
            elif second_half_avg < first_half_avg * 0.8:
                trend = 'improving'
        
        return {
            'status': 'active',
            'total_operations': len(operations),
            'avg_duration': avg_duration,
            'avg_memory_delta': avg_memory,
            'recent_avg_duration': recent_avg,
            'performance_trend': trend,
            'slow_operations_count': len(st.session_state.performance_state['slow_operations']),
            'active_alerts': len(st.session_state.performance_state['performance_alerts'])
        }
    
    def generate_optimization_suggestions(self) -> List[Dict[str, Any]]:
        """Generate optimization suggestions based on performance data"""
        suggestions = []
        operations = st.session_state.performance_state['operations']
        
        if not operations:
            return suggestions
        
        # Analyze operation patterns
        operation_stats = {}
        for op in operations:
            name = op['name']
            if name not in operation_stats:
                operation_stats[name] = {'durations': [], 'memory_deltas': []}
            
            operation_stats[name]['durations'].append(op['duration'])
            operation_stats[name]['memory_deltas'].append(op['memory_delta'])
        
        # Generate suggestions based on patterns
        for op_name, stats in operation_stats.items():
            avg_duration = sum(stats['durations']) / len(stats['durations'])
            avg_memory = sum(stats['memory_deltas']) / len(stats['memory_deltas'])
            
            # Slow operation suggestions
            if avg_duration > 10:  # Operations taking more than 10 seconds
                suggestions.append({
                    'type': 'performance',
                    'priority': 'high',
                    'operation': op_name,
                    'issue': 'slow_operation',
                    'message': f"{op_name} is slow (avg: {avg_duration:.1f}s)",
                    'suggestion': 'Consider implementing caching or optimizing the algorithm'
                })
            
            # High memory usage suggestions
            if avg_memory > 100:  # Operations using more than 100MB
                suggestions.append({
                    'type': 'memory',
                    'priority': 'medium',
                    'operation': op_name,
                    'issue': 'high_memory',
                    'message': f"{op_name} uses high memory (avg: {avg_memory:.1f}MB)",
                    'suggestion': 'Consider processing in smaller chunks or streaming'
                })
        
        return suggestions

class ProductionHardening:
    """Main production hardening coordinator"""
    
    def __init__(self):
        self.memory_manager = MemoryManager()
        self.network_resilience = NetworkResilience()
        self.performance_monitor = PerformanceMonitor()
        
        # Initialize session state for production monitoring
        if 'production_state' not in st.session_state:
            st.session_state.production_state = {
                'hardening_enabled': True,
                'monitoring_level': 'normal',  # minimal, normal, verbose
                'auto_cleanup': True,
                'performance_alerts': True,
                'health_checks': [],
                'system_status': 'healthy'
            }
        
        self.logger = logging.getLogger(__name__)
        self.setup_monitoring()
    
    def setup_monitoring(self):
        """Setup production monitoring"""
        # Register memory cleanup callbacks
        self.memory_manager.register_cleanup_callback(self.cleanup_session_state)
        self.memory_manager.register_cleanup_callback(self.cleanup_temporary_files)
        
        # Start background monitoring if enabled
        if st.session_state.production_state['hardening_enabled']:
            self.start_health_monitoring()
    
    def cleanup_session_state(self):
        """Cleanup session state to free memory"""
        # Clear large data structures from session state
        keys_to_clean = []
        
        for key, value in st.session_state.items():
            if isinstance(value, (list, dict)) and sys.getsizeof(value) > 1024 * 1024:  # > 1MB
                keys_to_clean.append(key)
        
        for key in keys_to_clean:
            if key not in ['production_state', 'memory_state', 'network_state', 'performance_state']:
                # Keep a backup of critical data
                if 'backup_' + key not in st.session_state:
                    st.session_state['backup_' + key] = st.session_state[key][-10:] if isinstance(st.session_state[key], list) else {}
                
                del st.session_state[key]
                self.logger.info(f"Cleaned up session state key: {key}")
    
    def cleanup_temporary_files(self):
        """Cleanup temporary files"""
        temp_dirs = ['/tmp', '/var/tmp']
        
        for temp_dir in temp_dirs:
            if os.path.exists(temp_dir):
                try:
                    # Clean files older than 1 hour
                    cutoff_time = time.time() - 3600
                    
                    for filename in os.listdir(temp_dir):
                        filepath = os.path.join(temp_dir, filename)
                        if os.path.isfile(filepath) and os.path.getmtime(filepath) < cutoff_time:
                            if filename.startswith('streamlit') or filename.startswith('tmp'):
                                os.remove(filepath)
                                self.logger.info(f"Cleaned up temp file: {filepath}")
                
                except Exception as e:
                    self.logger.error(f"Error cleaning temp files: {e}")
    
    def start_health_monitoring(self):
        """Start background health monitoring"""
        def health_check():
            while st.session_state.production_state['hardening_enabled']:
                try:
                    # Perform health checks
                    memory_usage = self.memory_manager.get_memory_usage()
                    network_health = self.network_resilience.get_network_health()
                    performance_summary = self.performance_monitor.get_performance_summary()
                    
                    # Update system status
                    status = self.determine_system_status(memory_usage, network_health, performance_summary)
                    st.session_state.production_state['system_status'] = status
                    
                    # Record health check
                    health_record = {
                        'timestamp': datetime.now().isoformat(),
                        'memory_usage': memory_usage,
                        'network_health': network_health,
                        'performance_summary': performance_summary,
                        'system_status': status
                    }
                    
                    st.session_state.production_state['health_checks'].append(health_record)
                    
                    # Keep only recent health checks (last 50)
                    if len(st.session_state.production_state['health_checks']) > 50:
                        st.session_state.production_state['health_checks'] = \
                            st.session_state.production_state['health_checks'][-50:]
                    
                    # Auto cleanup if needed
                    if st.session_state.production_state['auto_cleanup']:
                        if memory_usage['system_percent'] > 80:
                            self.memory_manager.cleanup_memory()
                    
                    time.sleep(30)  # Check every 30 seconds
                    
                except Exception as e:
                    self.logger.error(f"Health monitoring error: {e}")
                    time.sleep(60)  # Wait longer on error
        
        # Start monitoring thread (simplified for Streamlit)
        # Note: In production, this would be a proper background service
        pass
    
    def determine_system_status(self, memory_usage: Dict, network_health: Dict, performance_summary: Dict) -> str:
        """Determine overall system status"""
        issues = []
        
        # Check memory
        if memory_usage['system_percent'] > 90:
            issues.append('critical_memory')
        elif memory_usage['system_percent'] > 80:
            issues.append('high_memory')
        
        # Check network
        if network_health['health_status'] == 'poor':
            issues.append('network_issues')
        elif network_health['health_status'] == 'degraded':
            issues.append('network_degraded')
        
        # Check performance
        if performance_summary.get('performance_trend') == 'degrading':
            issues.append('performance_degrading')
        
        if performance_summary.get('slow_operations_count', 0) > 5:
            issues.append('slow_operations')
        
        # Determine status
        if any(issue.startswith('critical') for issue in issues):
            return 'critical'
        elif len(issues) >= 3:
            return 'degraded'
        elif len(issues) >= 1:
            return 'warning'
        else:
            return 'healthy'
    
    @contextmanager
    def production_operation(self, operation_name: str, expected_duration: Optional[float] = None):
        """Context manager for production-hardened operations"""
        with self.memory_manager.memory_monitor(operation_name):
            with self.network_resilience.network_operation(operation_name):
                with self.performance_monitor.performance_tracker(operation_name, expected_duration):
                    yield
    
    def show_production_dashboard(self):
        """Show production monitoring dashboard"""
        st.header("🛡️ Production Monitoring Dashboard")
        
        # System status overview
        status = st.session_state.production_state['system_status']
        status_colors = {
            'healthy': '🟢',
            'warning': '🟡',
            'degraded': '🟠',
            'critical': '🔴'
        }
        
        st.subheader(f"{status_colors.get(status, '⚪')} System Status: {status.title()}")
        
        # Key metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("💾 Memory")
            memory_usage = self.memory_manager.get_memory_usage()
            st.metric("System Memory", f"{memory_usage['system_percent']:.1f}%")
            st.metric("Process Memory", f"{memory_usage['process_mb']:.1f}MB")
            
            if memory_usage['system_percent'] > 80:
                st.warning("High memory usage detected")
                if st.button("🧹 Cleanup Memory"):
                    self.memory_manager.cleanup_memory(force=True)
                    st.success("Memory cleanup completed")
        
        with col2:
            st.subheader("🌐 Network")
            network_health = self.network_resilience.get_network_health()
            st.metric("Success Rate", f"{network_health['success_rate']:.1%}")
            st.metric("Total Requests", network_health['total_requests'])
            
            health_status = network_health['health_status']
            if health_status != 'good':
                st.warning(f"Network health: {health_status}")
        
        with col3:
            st.subheader("⚡ Performance")
            performance_summary = self.performance_monitor.get_performance_summary()
            
            if performance_summary['status'] == 'active':
                st.metric("Avg Duration", f"{performance_summary['avg_duration']:.2f}s")
                st.metric("Operations", performance_summary['total_operations'])
                
                trend = performance_summary['performance_trend']
                if trend != 'stable':
                    st.warning(f"Performance trend: {trend}")
            else:
                st.info("No performance data available")
        
        # Detailed monitoring
        with st.expander("📊 Detailed Monitoring", expanded=False):
            tab1, tab2, tab3 = st.tabs(["Memory Details", "Network Details", "Performance Details"])
            
            with tab1:
                self.show_memory_details()
            
            with tab2:
                self.show_network_details()
            
            with tab3:
                self.show_performance_details()
        
        # Production settings
        with st.expander("⚙️ Production Settings", expanded=False):
            self.show_production_settings()
    
    def show_memory_details(self):
        """Show detailed memory information"""
        st.subheader("Memory Management")
        
        memory_state = st.session_state.memory_state
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Peak Usage", f"{memory_state['peak_usage']:.1f}MB")
            st.metric("Cleanup Count", memory_state['cleanup_count'])
            st.metric("Warnings Issued", memory_state['warnings_issued'])
        
        with col2:
            if memory_state['last_cleanup']:
                last_cleanup = datetime.fromisoformat(memory_state['last_cleanup'])
                st.metric("Last Cleanup", last_cleanup.strftime("%H:%M:%S"))
            
            # Memory threshold setting
            new_threshold = st.slider(
                "Memory Threshold",
                min_value=0.5,
                max_value=0.95,
                value=self.memory_manager.memory_threshold,
                step=0.05,
                help="Trigger cleanup when memory usage exceeds this threshold"
            )
            self.memory_manager.memory_threshold = new_threshold
    
    def show_network_details(self):
        """Show detailed network information"""
        st.subheader("Network Resilience")
        
        network_state = st.session_state.network_state
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Failed Requests", network_state['failed_requests'])
            st.metric("Timeout Count", network_state['timeout_count'])
            st.metric("Retry Count", network_state['retry_count'])
        
        with col2:
            if network_state['last_failure']:
                last_failure = datetime.fromisoformat(network_state['last_failure'])
                st.metric("Last Failure", last_failure.strftime("%H:%M:%S"))
        
        # Recent connection issues
        issues = network_state['connection_issues']
        if issues:
            st.subheader("Recent Connection Issues")
            recent_issues = issues[-5:]  # Show last 5 issues
            
            for issue in recent_issues:
                timestamp = datetime.fromisoformat(issue['timestamp'])
                st.write(f"**{timestamp.strftime('%H:%M:%S')}** - {issue['operation']}: {issue['error_type']}")
    
    def show_performance_details(self):
        """Show detailed performance information"""
        st.subheader("Performance Monitoring")
        
        performance_state = st.session_state.performance_state
        
        # Recent operations
        operations = performance_state['operations']
        if operations:
            st.subheader("Recent Operations")
            recent_ops = operations[-10:]  # Show last 10 operations
            
            for op in recent_ops:
                timestamp = datetime.fromisoformat(op['timestamp'])
                duration_color = "🔴" if op['duration'] > 10 else "🟡" if op['duration'] > 5 else "🟢"
                st.write(f"{duration_color} **{op['name']}** - {op['duration']:.2f}s ({timestamp.strftime('%H:%M:%S')})")
        
        # Slow operations
        slow_ops = performance_state['slow_operations']
        if slow_ops:
            st.subheader("Slow Operations")
            for slow_op in slow_ops[-5:]:  # Show last 5 slow operations
                timestamp = datetime.fromisoformat(slow_op['timestamp'])
                st.warning(f"**{slow_op['operation']}** took {slow_op['duration']:.1f}s "
                          f"(expected {slow_op['expected']:.1f}s) - {timestamp.strftime('%H:%M:%S')}")
        
        # Optimization suggestions
        suggestions = self.performance_monitor.generate_optimization_suggestions()
        if suggestions:
            st.subheader("💡 Optimization Suggestions")
            for suggestion in suggestions:
                priority_icon = "🔴" if suggestion['priority'] == 'high' else "🟡" if suggestion['priority'] == 'medium' else "🔵"
                st.write(f"{priority_icon} **{suggestion['operation']}**: {suggestion['suggestion']}")
    
    def show_production_settings(self):
        """Show production configuration settings"""
        st.subheader("Production Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Hardening settings
            hardening_enabled = st.checkbox(
                "Production Hardening Enabled",
                value=st.session_state.production_state['hardening_enabled'],
                help="Enable production hardening features"
            )
            st.session_state.production_state['hardening_enabled'] = hardening_enabled
            
            auto_cleanup = st.checkbox(
                "Auto Memory Cleanup",
                value=st.session_state.production_state['auto_cleanup'],
                help="Automatically cleanup memory when usage is high"
            )
            st.session_state.production_state['auto_cleanup'] = auto_cleanup
        
        with col2:
            # Monitoring settings
            monitoring_level = st.selectbox(
                "Monitoring Level",
                options=['minimal', 'normal', 'verbose'],
                index=['minimal', 'normal', 'verbose'].index(st.session_state.production_state['monitoring_level']),
                help="Level of monitoring detail"
            )
            st.session_state.production_state['monitoring_level'] = monitoring_level
            
            performance_alerts = st.checkbox(
                "Performance Alerts",
                value=st.session_state.production_state['performance_alerts'],
                help="Show alerts for performance issues"
            )
            st.session_state.production_state['performance_alerts'] = performance_alerts
        
        # System actions
        st.subheader("System Actions")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔄 Force Memory Cleanup"):
                self.memory_manager.cleanup_memory(force=True)
                st.success("Memory cleanup completed")
        
        with col2:
            if st.button("📊 Generate Health Report"):
                self.generate_health_report()
        
        with col3:
            if st.button("🧹 Clear Monitoring Data"):
                self.clear_monitoring_data()
                st.success("Monitoring data cleared")
    
    def generate_health_report(self):
        """Generate comprehensive health report"""
        st.subheader("🏥 System Health Report")
        
        # Generate timestamp
        report_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.write(f"**Generated:** {report_time}")
        
        # System overview
        memory_usage = self.memory_manager.get_memory_usage()
        network_health = self.network_resilience.get_network_health()
        performance_summary = self.performance_monitor.get_performance_summary()
        
        # Health score calculation
        memory_score = max(0, 100 - memory_usage['system_percent'])
        network_score = network_health['success_rate'] * 100
        performance_score = 100 if performance_summary['status'] != 'active' else \
                          max(0, 100 - (performance_summary.get('slow_operations_count', 0) * 10))
        
        overall_score = (memory_score + network_score + performance_score) / 3
        
        # Display health score
        score_color = "🟢" if overall_score > 80 else "🟡" if overall_score > 60 else "🔴"
        st.metric("Overall Health Score", f"{score_color} {overall_score:.1f}/100")
        
        # Detailed breakdown
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Memory Health", f"{memory_score:.1f}/100")
        with col2:
            st.metric("Network Health", f"{network_score:.1f}/100")
        with col3:
            st.metric("Performance Health", f"{performance_score:.1f}/100")
        
        # Recommendations
        recommendations = []
        
        if memory_score < 70:
            recommendations.append("🔴 High memory usage - consider cleanup or optimization")
        if network_score < 80:
            recommendations.append("🟡 Network issues detected - check connectivity")
        if performance_score < 70:
            recommendations.append("🟡 Performance issues - review slow operations")
        
        if recommendations:
            st.subheader("🎯 Recommendations")
            for rec in recommendations:
                st.write(rec)
        else:
            st.success("✅ System is operating optimally")
    
    def clear_monitoring_data(self):
        """Clear all monitoring data"""
        # Clear session state monitoring data
        st.session_state.memory_state = {
            'peak_usage': 0,
            'cleanup_count': 0,
            'warnings_issued': 0,
            'last_cleanup': None
        }
        
        st.session_state.network_state = {
            'total_requests': 0,
            'failed_requests': 0,
            'timeout_count': 0,
            'retry_count': 0,
            'last_failure': None,
            'connection_issues': []
        }
        
        st.session_state.performance_state = {
            'operations': [],
            'slow_operations': [],
            'performance_alerts': [],
            'baseline_metrics': {},
            'optimization_suggestions': []
        }
        
        st.session_state.production_state['health_checks'] = []
        
        # Clear internal data
        self.performance_monitor.performance_history = []
        self.performance_monitor.operation_benchmarks = {}

# Global instance
production_hardening = ProductionHardening()

