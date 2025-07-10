#!/usr/bin/env python3
"""
Real-Time Quality Assurance Dashboard
Live quality monitoring, automatic threshold routing, and smart recommendations
"""

import time
import statistics
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import streamlit as st
import pandas as pd
import logging

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

class QualityThreshold:
    """Quality threshold configuration"""
    def __init__(self, name: str, min_score: float, max_score: float, action: str, color: str):
        self.name = name
        self.min_score = min_score
        self.max_score = max_score
        self.action = action
        self.color = color

class RealtimeQualityDashboard:
    """Real-time quality monitoring and assurance system"""
    
    def __init__(self):
        self.quality_history = []
        self.processing_metrics = {}
        self.alert_thresholds = {}
        
        # Initialize session state for quality tracking
        if 'quality_dashboard_state' not in st.session_state:
            st.session_state.quality_dashboard_state = {
                'quality_samples': [],
                'processing_stats': {},
                'alerts': [],
                'recommendations': [],
                'auto_routing_enabled': True,
                'quality_trends': {},
                'last_update': None
            }
        
        self.logger = logging.getLogger(__name__)
        self.setup_quality_thresholds()
        self.setup_alert_system()
    
    def setup_quality_thresholds(self):
        """Configure quality thresholds for automatic routing"""
        self.quality_thresholds = {
            'excellent': QualityThreshold('Excellent', 0.9, 1.0, 'auto_approve', '#28a745'),
            'very_good': QualityThreshold('Very Good', 0.8, 0.9, 'auto_approve', '#20c997'),
            'good': QualityThreshold('Good', 0.7, 0.8, 'auto_approve', '#ffc107'),
            'fair': QualityThreshold('Fair', 0.6, 0.7, 'manual_review', '#fd7e14'),
            'poor': QualityThreshold('Poor', 0.4, 0.6, 'manual_review', '#dc3545'),
            'very_poor': QualityThreshold('Very Poor', 0.0, 0.4, 'reject', '#6c757d')
        }
        
        # User-configurable thresholds
        if 'quality_thresholds_config' not in st.session_state:
            st.session_state.quality_thresholds_config = {
                'auto_approve_threshold': 0.7,
                'manual_review_threshold': 0.4,
                'reject_threshold': 0.3,
                'batch_quality_threshold': 0.6,
                'consistency_threshold': 0.15  # Standard deviation threshold
            }
    
    def setup_alert_system(self):
        """Configure alert system for quality monitoring"""
        self.alert_configs = {
            'quality_drop': {
                'threshold': 0.2,  # 20% drop in average quality
                'window': 10,      # Last 10 samples
                'severity': 'high',
                'message': 'Significant quality drop detected'
            },
            'consistency_issue': {
                'threshold': 0.25,  # High standard deviation
                'window': 15,
                'severity': 'medium',
                'message': 'Quality consistency issues detected'
            },
            'low_batch_quality': {
                'threshold': 0.5,   # Batch average below 50%
                'window': 5,
                'severity': 'high',
                'message': 'Low batch quality detected'
            },
            'processing_slowdown': {
                'threshold': 2.0,   # 2x slower than average
                'window': 5,
                'severity': 'medium',
                'message': 'Processing performance degradation'
            }
        }
    
    def update_quality_metrics(self, content_id: str, quality_data: Dict[str, Any]):
        """Update quality metrics with new data"""
        timestamp = datetime.now()
        
        quality_sample = {
            'content_id': content_id,
            'timestamp': timestamp.isoformat(),
            'quality_score': quality_data.get('quality_score', 0),
            'quality_grade': quality_data.get('quality_grade', 'Unknown'),
            'word_count': quality_data.get('word_count', 0),
            'readability_score': quality_data.get('readability_score', 0),
            'coherence_score': quality_data.get('coherence_score', 0),
            'dialogue_potential': quality_data.get('dialogue_potential', 0),
            'processing_time': quality_data.get('processing_time', 0),
            'issues_count': len(quality_data.get('issues', [])),
            'warnings_count': len(quality_data.get('warnings', []))
        }
        
        # Add to session state
        st.session_state.quality_dashboard_state['quality_samples'].append(quality_sample)
        
        # Keep only recent samples (last 100)
        if len(st.session_state.quality_dashboard_state['quality_samples']) > 100:
            st.session_state.quality_dashboard_state['quality_samples'] = \
                st.session_state.quality_dashboard_state['quality_samples'][-100:]
        
        # Update processing stats
        self.update_processing_stats(quality_sample)
        
        # Check for alerts
        self.check_quality_alerts()
        
        # Generate recommendations
        self.generate_smart_recommendations(quality_sample)
        
        # Update last update time
        st.session_state.quality_dashboard_state['last_update'] = timestamp.isoformat()
        
        self.logger.info(f"Quality metrics updated for {content_id}: {quality_sample['quality_score']:.2f}")
    
    def update_processing_stats(self, quality_sample: Dict[str, Any]):
        """Update processing statistics"""
        stats = st.session_state.quality_dashboard_state['processing_stats']
        
        # Update counters
        stats['total_processed'] = stats.get('total_processed', 0) + 1
        stats['total_words'] = stats.get('total_words', 0) + quality_sample['word_count']
        stats['total_processing_time'] = stats.get('total_processing_time', 0) + quality_sample['processing_time']
        
        # Update quality distribution
        grade = quality_sample['quality_grade']
        if 'quality_distribution' not in stats:
            stats['quality_distribution'] = {}
        stats['quality_distribution'][grade] = stats['quality_distribution'].get(grade, 0) + 1
        
        # Update averages
        samples = st.session_state.quality_dashboard_state['quality_samples']
        if samples:
            stats['avg_quality'] = statistics.mean([s['quality_score'] for s in samples])
            stats['avg_processing_time'] = statistics.mean([s['processing_time'] for s in samples])
            stats['avg_word_count'] = statistics.mean([s['word_count'] for s in samples])
            
            # Calculate quality consistency
            quality_scores = [s['quality_score'] for s in samples[-20:]]  # Last 20 samples
            if len(quality_scores) > 1:
                stats['quality_std'] = statistics.stdev(quality_scores)
                stats['quality_consistency'] = 1 - min(1, stats['quality_std'] / 0.5)  # Normalize to 0-1
    
    def check_quality_alerts(self):
        """Check for quality alerts and issues"""
        samples = st.session_state.quality_dashboard_state['quality_samples']
        alerts = st.session_state.quality_dashboard_state['alerts']
        
        if len(samples) < 5:  # Need minimum samples
            return
        
        current_time = datetime.now()
        
        # Check quality drop alert
        self.check_quality_drop_alert(samples, alerts, current_time)
        
        # Check consistency alert
        self.check_consistency_alert(samples, alerts, current_time)
        
        # Check batch quality alert
        self.check_batch_quality_alert(samples, alerts, current_time)
        
        # Check processing performance alert
        self.check_processing_performance_alert(samples, alerts, current_time)
        
        # Clean old alerts (older than 1 hour)
        cutoff_time = current_time - timedelta(hours=1)
        st.session_state.quality_dashboard_state['alerts'] = [
            alert for alert in alerts 
            if datetime.fromisoformat(alert['timestamp']) > cutoff_time
        ]
    
    def check_quality_drop_alert(self, samples: List[Dict], alerts: List[Dict], current_time: datetime):
        """Check for significant quality drops"""
        config = self.alert_configs['quality_drop']
        window_size = config['window']
        
        if len(samples) < window_size * 2:
            return
        
        # Compare recent window with previous window
        recent_scores = [s['quality_score'] for s in samples[-window_size:]]
        previous_scores = [s['quality_score'] for s in samples[-window_size*2:-window_size]]
        
        recent_avg = statistics.mean(recent_scores)
        previous_avg = statistics.mean(previous_scores)
        
        drop_percentage = (previous_avg - recent_avg) / previous_avg if previous_avg > 0 else 0
        
        if drop_percentage > config['threshold']:
            alert = {
                'type': 'quality_drop',
                'severity': config['severity'],
                'message': f"{config['message']}: {drop_percentage:.1%} drop",
                'timestamp': current_time.isoformat(),
                'data': {
                    'recent_avg': recent_avg,
                    'previous_avg': previous_avg,
                    'drop_percentage': drop_percentage
                }
            }
            alerts.append(alert)
            self.logger.warning(f"Quality drop alert: {drop_percentage:.1%}")
    
    def check_consistency_alert(self, samples: List[Dict], alerts: List[Dict], current_time: datetime):
        """Check for quality consistency issues"""
        config = self.alert_configs['consistency_issue']
        window_size = config['window']
        
        if len(samples) < window_size:
            return
        
        recent_scores = [s['quality_score'] for s in samples[-window_size:]]
        std_dev = statistics.stdev(recent_scores) if len(recent_scores) > 1 else 0
        
        if std_dev > config['threshold']:
            alert = {
                'type': 'consistency_issue',
                'severity': config['severity'],
                'message': f"{config['message']}: σ={std_dev:.2f}",
                'timestamp': current_time.isoformat(),
                'data': {
                    'std_dev': std_dev,
                    'scores': recent_scores
                }
            }
            alerts.append(alert)
            self.logger.warning(f"Consistency alert: σ={std_dev:.2f}")
    
    def check_batch_quality_alert(self, samples: List[Dict], alerts: List[Dict], current_time: datetime):
        """Check for low batch quality"""
        config = self.alert_configs['low_batch_quality']
        window_size = config['window']
        
        if len(samples) < window_size:
            return
        
        recent_scores = [s['quality_score'] for s in samples[-window_size:]]
        batch_avg = statistics.mean(recent_scores)
        
        if batch_avg < config['threshold']:
            alert = {
                'type': 'low_batch_quality',
                'severity': config['severity'],
                'message': f"{config['message']}: {batch_avg:.2f} average",
                'timestamp': current_time.isoformat(),
                'data': {
                    'batch_avg': batch_avg,
                    'threshold': config['threshold']
                }
            }
            alerts.append(alert)
            self.logger.warning(f"Low batch quality alert: {batch_avg:.2f}")
    
    def check_processing_performance_alert(self, samples: List[Dict], alerts: List[Dict], current_time: datetime):
        """Check for processing performance issues"""
        config = self.alert_configs['processing_slowdown']
        window_size = config['window']
        
        if len(samples) < window_size * 2:
            return
        
        recent_times = [s['processing_time'] for s in samples[-window_size:]]
        all_times = [s['processing_time'] for s in samples]
        
        recent_avg = statistics.mean(recent_times)
        overall_avg = statistics.mean(all_times)
        
        slowdown_factor = recent_avg / overall_avg if overall_avg > 0 else 1
        
        if slowdown_factor > config['threshold']:
            alert = {
                'type': 'processing_slowdown',
                'severity': config['severity'],
                'message': f"{config['message']}: {slowdown_factor:.1f}x slower",
                'timestamp': current_time.isoformat(),
                'data': {
                    'recent_avg': recent_avg,
                    'overall_avg': overall_avg,
                    'slowdown_factor': slowdown_factor
                }
            }
            alerts.append(alert)
            self.logger.warning(f"Performance alert: {slowdown_factor:.1f}x slower")
    
    def generate_smart_recommendations(self, quality_sample: Dict[str, Any]):
        """Generate smart recommendations based on quality analysis"""
        recommendations = st.session_state.quality_dashboard_state['recommendations']
        
        quality_score = quality_sample['quality_score']
        quality_grade = quality_sample['quality_grade']
        
        # Clear old recommendations for this content
        content_id = quality_sample['content_id']
        recommendations = [r for r in recommendations if r.get('content_id') != content_id]
        
        # Generate new recommendations
        new_recommendations = []
        
        # Quality-based recommendations
        if quality_score < 0.4:
            new_recommendations.append({
                'type': 'quality_improvement',
                'priority': 'high',
                'message': 'Content quality is very low - consider manual review or re-processing',
                'action': 'manual_review',
                'content_id': content_id
            })
        elif quality_score < 0.6:
            new_recommendations.append({
                'type': 'quality_improvement',
                'priority': 'medium',
                'message': 'Content quality could be improved - try different enhancement settings',
                'action': 'enhance_settings',
                'content_id': content_id
            })
        
        # Specific metric recommendations
        if quality_sample['coherence_score'] < 0.5:
            new_recommendations.append({
                'type': 'coherence',
                'priority': 'medium',
                'message': 'Low coherence detected - consider restructuring content',
                'action': 'restructure',
                'content_id': content_id
            })
        
        if quality_sample['dialogue_potential'] < 0.3:
            new_recommendations.append({
                'type': 'dialogue',
                'priority': 'low',
                'message': 'Low dialogue potential - try narrative-to-dialogue enhancement',
                'action': 'dialogue_enhancement',
                'content_id': content_id
            })
        
        if quality_sample['word_count'] < 50:
            new_recommendations.append({
                'type': 'length',
                'priority': 'medium',
                'message': 'Content too short - consider combining with other chunks',
                'action': 'combine_chunks',
                'content_id': content_id
            })
        elif quality_sample['word_count'] > 800:
            new_recommendations.append({
                'type': 'length',
                'priority': 'low',
                'message': 'Content very long - consider splitting into smaller chunks',
                'action': 'split_chunk',
                'content_id': content_id
            })
        
        # Add new recommendations
        recommendations.extend(new_recommendations)
        
        # Keep only recent recommendations (last 50)
        if len(recommendations) > 50:
            recommendations = recommendations[-50:]
        
        st.session_state.quality_dashboard_state['recommendations'] = recommendations
    
    def route_content_automatically(self, content_id: str, quality_score: float) -> str:
        """Automatically route content based on quality thresholds"""
        if not st.session_state.quality_dashboard_state['auto_routing_enabled']:
            return 'manual_review'
        
        config = st.session_state.quality_thresholds_config
        
        if quality_score >= config['auto_approve_threshold']:
            self.logger.info(f"Auto-approved content {content_id}: {quality_score:.2f}")
            return 'auto_approve'
        elif quality_score >= config['manual_review_threshold']:
            self.logger.info(f"Routed to manual review {content_id}: {quality_score:.2f}")
            return 'manual_review'
        else:
            self.logger.info(f"Auto-rejected content {content_id}: {quality_score:.2f}")
            return 'reject'
    
    def show_realtime_dashboard(self):
        """Display the real-time quality dashboard"""
        st.header("📊 Real-Time Quality Dashboard")
        
        # Dashboard controls
        col1, col2, col3 = st.columns(3)
        
        with col1:
            auto_routing = st.checkbox(
                "Auto-routing Enabled",
                value=st.session_state.quality_dashboard_state['auto_routing_enabled'],
                help="Automatically route content based on quality thresholds"
            )
            st.session_state.quality_dashboard_state['auto_routing_enabled'] = auto_routing
        
        with col2:
            if st.button("🔄 Refresh Dashboard"):
                st.rerun()
        
        with col3:
            if st.button("🧹 Clear History"):
                st.session_state.quality_dashboard_state['quality_samples'] = []
                st.session_state.quality_dashboard_state['alerts'] = []
                st.success("Dashboard cleared!")
        
        # Key metrics overview
        self.show_key_metrics()
        
        # Quality trends
        self.show_quality_trends()
        
        # Alerts and recommendations
        col1, col2 = st.columns(2)
        
        with col1:
            self.show_alerts_panel()
        
        with col2:
            self.show_recommendations_panel()
        
        # Detailed analytics
        with st.expander("📈 Detailed Analytics", expanded=False):
            self.show_detailed_analytics()
        
        # Quality threshold configuration
        with st.expander("⚙️ Quality Threshold Configuration", expanded=False):
            self.show_threshold_configuration()
    
    def show_key_metrics(self):
        """Show key quality metrics"""
        stats = st.session_state.quality_dashboard_state['processing_stats']
        
        if not stats:
            st.info("No quality data available yet. Process some content to see metrics.")
            return
        
        # Main metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric(
                "Total Processed",
                stats.get('total_processed', 0)
            )
        
        with col2:
            avg_quality = stats.get('avg_quality', 0)
            st.metric(
                "Avg Quality",
                f"{avg_quality:.2f}",
                delta=f"{(avg_quality - 0.7):.2f}" if avg_quality > 0 else None
            )
        
        with col3:
            consistency = stats.get('quality_consistency', 0)
            st.metric(
                "Consistency",
                f"{consistency:.2f}",
                delta=f"{(consistency - 0.8):.2f}" if consistency > 0 else None
            )
        
        with col4:
            avg_time = stats.get('avg_processing_time', 0)
            st.metric(
                "Avg Time",
                f"{avg_time:.1f}s"
            )
        
        with col5:
            total_words = stats.get('total_words', 0)
            st.metric(
                "Total Words",
                f"{total_words:,}"
            )
    
    def show_quality_trends(self):
        """Show quality trends over time"""
        samples = st.session_state.quality_dashboard_state['quality_samples']
        
        if len(samples) < 2:
            st.info("Need more data points to show trends.")
            return
        
        st.subheader("📈 Quality Trends")
        
        if PLOTLY_AVAILABLE:
            # Create quality trend chart
            df = pd.DataFrame(samples)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Quality Score Over Time', 'Processing Time', 
                              'Quality Distribution', 'Metrics Correlation'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"type": "pie"}, {"type": "scatter"}]]
            )
            
            # Quality score trend
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=df['quality_score'],
                    mode='lines+markers',
                    name='Quality Score',
                    line=dict(color='#1f77b4')
                ),
                row=1, col=1
            )
            
            # Processing time trend
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=df['processing_time'],
                    mode='lines+markers',
                    name='Processing Time',
                    line=dict(color='#ff7f0e')
                ),
                row=1, col=2
            )
            
            # Quality distribution pie chart
            quality_counts = df['quality_grade'].value_counts()
            fig.add_trace(
                go.Pie(
                    labels=quality_counts.index,
                    values=quality_counts.values,
                    name="Quality Distribution"
                ),
                row=2, col=1
            )
            
            # Correlation scatter plot
            fig.add_trace(
                go.Scatter(
                    x=df['word_count'],
                    y=df['quality_score'],
                    mode='markers',
                    name='Quality vs Word Count',
                    marker=dict(
                        color=df['processing_time'],
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title="Processing Time")
                    )
                ),
                row=2, col=2
            )
            
            fig.update_layout(
                height=600,
                showlegend=True,
                title_text="Quality Analytics Dashboard"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        else:
            # Fallback to simple line chart
            df = pd.DataFrame(samples)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            st.line_chart(df.set_index('timestamp')['quality_score'])
    
    def show_alerts_panel(self):
        """Show active alerts"""
        st.subheader("🚨 Active Alerts")
        
        alerts = st.session_state.quality_dashboard_state['alerts']
        
        if not alerts:
            st.success("✅ No active alerts")
            return
        
        # Group alerts by severity
        high_alerts = [a for a in alerts if a['severity'] == 'high']
        medium_alerts = [a for a in alerts if a['severity'] == 'medium']
        low_alerts = [a for a in alerts if a['severity'] == 'low']
        
        # Show high severity alerts
        if high_alerts:
            st.error(f"🔴 High Priority ({len(high_alerts)})")
            for alert in high_alerts[-3:]:  # Show last 3
                st.write(f"• {alert['message']}")
        
        # Show medium severity alerts
        if medium_alerts:
            st.warning(f"🟡 Medium Priority ({len(medium_alerts)})")
            for alert in medium_alerts[-2:]:  # Show last 2
                st.write(f"• {alert['message']}")
        
        # Show low severity alerts
        if low_alerts:
            st.info(f"🔵 Low Priority ({len(low_alerts)})")
            for alert in low_alerts[-1:]:  # Show last 1
                st.write(f"• {alert['message']}")
    
    def show_recommendations_panel(self):
        """Show smart recommendations"""
        st.subheader("💡 Smart Recommendations")
        
        recommendations = st.session_state.quality_dashboard_state['recommendations']
        
        if not recommendations:
            st.success("✅ No recommendations")
            return
        
        # Group by priority
        high_priority = [r for r in recommendations if r['priority'] == 'high']
        medium_priority = [r for r in recommendations if r['priority'] == 'medium']
        low_priority = [r for r in recommendations if r['priority'] == 'low']
        
        # Show high priority recommendations
        if high_priority:
            st.error(f"🔴 High Priority ({len(high_priority)})")
            for rec in high_priority[-3:]:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"• {rec['message']}")
                with col2:
                    if st.button("Apply", key=f"apply_{rec.get('content_id', 'unknown')}_{rec['type']}"):
                        self.apply_recommendation(rec)
        
        # Show medium priority recommendations
        if medium_priority:
            st.warning(f"🟡 Medium Priority ({len(medium_priority)})")
            for rec in medium_priority[-2:]:
                st.write(f"• {rec['message']}")
        
        # Show low priority recommendations
        if low_priority:
            st.info(f"🔵 Low Priority ({len(low_priority)})")
            for rec in low_priority[-1:]:
                st.write(f"• {rec['message']}")
    
    def show_detailed_analytics(self):
        """Show detailed analytics and insights"""
        samples = st.session_state.quality_dashboard_state['quality_samples']
        
        if len(samples) < 5:
            st.info("Need more data for detailed analytics.")
            return
        
        df = pd.DataFrame(samples)
        
        # Statistical summary
        st.subheader("📊 Statistical Summary")
        
        numeric_columns = ['quality_score', 'word_count', 'readability_score', 
                          'coherence_score', 'dialogue_potential', 'processing_time']
        
        summary_stats = df[numeric_columns].describe()
        st.dataframe(summary_stats)
        
        # Correlation analysis
        st.subheader("🔗 Correlation Analysis")
        correlation_matrix = df[numeric_columns].corr()
        
        if PLOTLY_AVAILABLE:
            fig = px.imshow(
                correlation_matrix,
                text_auto=True,
                aspect="auto",
                title="Metric Correlations"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.dataframe(correlation_matrix)
        
        # Quality patterns
        st.subheader("🔍 Quality Patterns")
        
        # Time-based patterns
        df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
        hourly_quality = df.groupby('hour')['quality_score'].mean()
        
        st.write("**Quality by Hour of Day:**")
        st.bar_chart(hourly_quality)
        
        # Word count vs quality
        st.write("**Quality vs Word Count Analysis:**")
        word_count_bins = pd.cut(df['word_count'], bins=5, labels=['Very Short', 'Short', 'Medium', 'Long', 'Very Long'])
        quality_by_length = df.groupby(word_count_bins)['quality_score'].mean()
        st.bar_chart(quality_by_length)
    
    def show_threshold_configuration(self):
        """Show quality threshold configuration interface"""
        st.subheader("⚙️ Quality Thresholds")
        
        config = st.session_state.quality_thresholds_config
        
        col1, col2 = st.columns(2)
        
        with col1:
            config['auto_approve_threshold'] = st.slider(
                "Auto-Approve Threshold",
                min_value=0.0,
                max_value=1.0,
                value=config['auto_approve_threshold'],
                step=0.05,
                help="Content above this score is automatically approved"
            )
            
            config['manual_review_threshold'] = st.slider(
                "Manual Review Threshold",
                min_value=0.0,
                max_value=1.0,
                value=config['manual_review_threshold'],
                step=0.05,
                help="Content above this score goes to manual review"
            )
        
        with col2:
            config['reject_threshold'] = st.slider(
                "Reject Threshold",
                min_value=0.0,
                max_value=1.0,
                value=config['reject_threshold'],
                step=0.05,
                help="Content below this score is automatically rejected"
            )
            
            config['consistency_threshold'] = st.slider(
                "Consistency Threshold",
                min_value=0.0,
                max_value=0.5,
                value=config['consistency_threshold'],
                step=0.01,
                help="Standard deviation threshold for consistency alerts"
            )
        
        # Show current routing distribution
        if st.session_state.quality_dashboard_state['quality_samples']:
            st.write("**Current Routing Distribution:**")
            samples = st.session_state.quality_dashboard_state['quality_samples']
            
            auto_approve = sum(1 for s in samples if s['quality_score'] >= config['auto_approve_threshold'])
            manual_review = sum(1 for s in samples if config['manual_review_threshold'] <= s['quality_score'] < config['auto_approve_threshold'])
            reject = sum(1 for s in samples if s['quality_score'] < config['manual_review_threshold'])
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Auto-Approve", auto_approve, delta=f"{auto_approve/len(samples)*100:.1f}%")
            with col2:
                st.metric("Manual Review", manual_review, delta=f"{manual_review/len(samples)*100:.1f}%")
            with col3:
                st.metric("Reject", reject, delta=f"{reject/len(samples)*100:.1f}%")
    
    def apply_recommendation(self, recommendation: Dict[str, Any]):
        """Apply a recommendation action"""
        action = recommendation['action']
        content_id = recommendation.get('content_id')
        
        if action == 'manual_review':
            st.info(f"Content {content_id} moved to manual review queue")
        elif action == 'enhance_settings':
            st.info(f"Enhancement settings updated for {content_id}")
        elif action == 'restructure':
            st.info(f"Content {content_id} marked for restructuring")
        elif action == 'dialogue_enhancement':
            st.info(f"Dialogue enhancement applied to {content_id}")
        elif action == 'combine_chunks':
            st.info(f"Content {content_id} marked for combination")
        elif action == 'split_chunk':
            st.info(f"Content {content_id} marked for splitting")
        
        # Remove applied recommendation
        recommendations = st.session_state.quality_dashboard_state['recommendations']
        st.session_state.quality_dashboard_state['recommendations'] = [
            r for r in recommendations if r != recommendation
        ]
    
    def get_quality_summary(self) -> Dict[str, Any]:
        """Get summary of current quality status"""
        stats = st.session_state.quality_dashboard_state['processing_stats']
        alerts = st.session_state.quality_dashboard_state['alerts']
        recommendations = st.session_state.quality_dashboard_state['recommendations']
        
        return {
            'total_processed': stats.get('total_processed', 0),
            'avg_quality': stats.get('avg_quality', 0),
            'quality_consistency': stats.get('quality_consistency', 0),
            'active_alerts': len(alerts),
            'high_priority_alerts': len([a for a in alerts if a['severity'] == 'high']),
            'pending_recommendations': len(recommendations),
            'auto_routing_enabled': st.session_state.quality_dashboard_state['auto_routing_enabled'],
            'last_update': st.session_state.quality_dashboard_state['last_update']
        }

# Global instance
realtime_quality_dashboard = RealtimeQualityDashboard()

