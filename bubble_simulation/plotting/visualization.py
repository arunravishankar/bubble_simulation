import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
# from io import BytesIO
# import base64
import logging as logger
warnings.filterwarnings('ignore')

# Set style for publication-quality plots
plt.style.use('default')
sns.set_palette("husl")

class BubbleVisualizationSuite:
    """
    State-of-the-art visualization suite for filter bubble analysis.
    
    Creates publication-quality plots that tell the story of filter bubble
    formation and prevention in recommendation systems.
    """
    
    def __init__(self, figsize_default=(12, 8), style='whitegrid'):
        """
        Initialize the visualization suite.
        
        Args:
            figsize_default: Default figure size for matplotlib plots
            style: Seaborn style to use
        """
        self.figsize_default = figsize_default
        self.style = style
        sns.set_style(style)
        
        # Color schemes for different recommenders
        self.color_schemes = {
            'Traditional': '#E74C3C',      # Red - represents bubble formation
            'Collaborative': '#F39C12',    # Orange - middle ground
            'PopularityRecommender': '#F39C12',  # Orange - same as Collaborative
            'CollaborativeFilteringRecommender': '#F39C12',  # Orange
            'RL': '#27AE60',              # Green - represents bubble prevention
            'RLRecommender': '#27AE60',    # Green - same as RL
            'Baseline': '#95A5A6'          # Gray - neutral baseline
        }
        
        # Extended color palette for multiple systems
        self.extended_colors = [
            '#E74C3C', '#27AE60', '#F39C12', '#3498DB', '#9B59B6', 
            '#1ABC9C', '#E67E22', '#34495E', '#F1C40F', '#E91E63'
        ]
    
    def _save_plotly_as_png(self, fig, save_path: str, width: int = 800, height: int = 600):
        """Convert plotly figure to PNG using matplotlib backend."""
        try:
            # Try to use plotly's built-in static image export without kaleido
            fig.write_image(save_path, width=width, height=height)
            logger.info(f"Successfully saved PNG using plotly: {save_path}")
        except Exception as e:
            logger.warning(f"Plotly PNG export failed ({str(e)}), using matplotlib fallback")
            # Fallback: create a simple matplotlib version
            import matplotlib.pyplot as plt
            plt.figure(figsize=(width/100, height/100))
            
            # Extract data from plotly figure (simplified)
            if hasattr(fig, 'data') and len(fig.data) > 0:
                for trace in fig.data:
                    if hasattr(trace, 'x') and hasattr(trace, 'y') and trace.x is not None and trace.y is not None:
                        plt.plot(trace.x, trace.y, label=trace.name if hasattr(trace, 'name') else '')
            
            plt.title(fig.layout.title.text if hasattr(fig.layout, 'title') and fig.layout.title else 'Visualization')
            plt.legend()
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            logger.warning(f"Used matplotlib fallback for {save_path}")

    def create_bubble_formation_story(self, 
                                    results_dict: Dict[str, Any],
                                    save_path: Optional[str] = None) -> go.Figure:
        """
        Create the main story visualization showing filter bubble formation over time.
        
        This is the hero plot for your blog post - it clearly shows how
        traditional recommenders create bubbles while RL prevents them.
        
        Args:
            results_dict: Dictionary mapping recommender names to SimulationResults
            save_path: Path to save the plot
            
        Returns:
            Plotly figure object
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Content Diversity Over Time', 'Engagement Rate Over Time',
                'Business Impact: Catalog Coverage', 'User Satisfaction Proxy'
            ),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]],
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )
        
        # Plot 1: Content Diversity (the main story)
        for name, results in results_dict.items():
            color = self.color_schemes.get(name, self.extended_colors[len(fig.data) % len(self.extended_colors)])
            
            # Add diversity line with markers
            fig.add_trace(
                go.Scatter(
                    x=results.timesteps,
                    y=results.content_diversity,
                    mode='lines+markers',
                    name=f'{name}',
                    line=dict(color=color, width=3),
                    marker=dict(size=6),
                    hovertemplate=f'<b>{name}</b><br>' +
                                 'Timestep: %{x}<br>' +
                                 'Diversity: %{y:.3f}<br>' +
                                 '<extra></extra>'
                ),
                row=1, col=1
            )
        
        # Plot 2: Engagement Rate
        for name, results in results_dict.items():
            color = self.color_schemes.get(name, self.extended_colors[len(results_dict) % len(self.extended_colors)])
            
            fig.add_trace(
                go.Scatter(
                    x=results.timesteps,
                    y=results.engagement_rate,
                    mode='lines+markers',
                    name=f'{name}',
                    line=dict(color=color, width=3),
                    marker=dict(size=6),
                    showlegend=False,
                    hovertemplate=f'<b>{name}</b><br>' +
                                 'Timestep: %{x}<br>' +
                                 'Engagement: %{y:.3f}<br>' +
                                 '<extra></extra>'
                ),
                row=1, col=2
            )
        
        # Plot 3: Catalog Coverage
        for name, results in results_dict.items():
            color = self.color_schemes.get(name, self.extended_colors[len(results_dict) % len(self.extended_colors)])
            
            fig.add_trace(
                go.Scatter(
                    x=results.timesteps,
                    y=results.catalog_coverage,
                    mode='lines+markers',
                    name=f'{name}',
                    line=dict(color=color, width=3),
                    marker=dict(size=6),
                    showlegend=False,
                    hovertemplate=f'<b>{name}</b><br>' +
                                 'Timestep: %{x}<br>' +
                                 'Coverage: %{y:.3f}<br>' +
                                 '<extra></extra>'
                ),
                row=2, col=1
            )
        
        # Plot 4: User Diversity (satisfaction proxy)
        for name, results in results_dict.items():
            color = self.color_schemes.get(name, self.extended_colors[len(results_dict) % len(self.extended_colors)])
            
            fig.add_trace(
                go.Scatter(
                    x=results.timesteps,
                    y=results.avg_user_diversity,
                    mode='lines+markers',
                    name=f'{name}',
                    line=dict(color=color, width=3),
                    marker=dict(size=6),
                    showlegend=False,
                    hovertemplate=f'<b>{name}</b><br>' +
                                 'Timestep: %{x}<br>' +
                                 'User Diversity: %{y:.3f}<br>' +
                                 '<extra></extra>'
                ),
                row=2, col=2
            )
        
        # Add critical annotations
        # Highlight the "bubble formation zone"
        fig.add_vrect(
            x0=20, x1=40,
            fillcolor="rgba(255,0,0,0.1)",
            layer="below",
            line_width=0,
            row=1, col=1
        )
        
        # Update layout with professional styling
        fig.update_layout(
            title={
                'text': '<b>Filter Bubble Formation vs Prevention: A Tale of Two Approaches</b><br>' +
                       '<sub>How Traditional Recommenders Create Bubbles While RL Maintains Diversity</sub>',
                'x': 0.5,
                'font': {'size': 20, 'family': 'Arial, sans-serif'}
            },
            font=dict(family="Arial, sans-serif", size=12),
            plot_bgcolor='white',
            paper_bgcolor='white',
            height=800,
            width=1200,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                font=dict(size=14)
            )
        )
        
        # Update axes labels and formatting
        fig.update_xaxes(title_text="Simulation Timestep", row=1, col=1, title_font=dict(size=14))
        fig.update_xaxes(title_text="Simulation Timestep", row=1, col=2, title_font=dict(size=14))
        fig.update_xaxes(title_text="Simulation Timestep", row=2, col=1, title_font=dict(size=14))
        fig.update_xaxes(title_text="Simulation Timestep", row=2, col=2, title_font=dict(size=14))
        
        fig.update_yaxes(title_text="Content Diversity", row=1, col=1, title_font=dict(size=14))
        fig.update_yaxes(title_text="Engagement Rate", row=1, col=2, title_font=dict(size=14))
        fig.update_yaxes(title_text="Catalog Coverage", row=2, col=1, title_font=dict(size=14))
        fig.update_yaxes(title_text="User Diversity", row=2, col=2, title_font=dict(size=14))
        
        # Add grid
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        
        if save_path:
            fig.write_html(save_path)
            self._save_plotly_as_png(fig, save_path.replace('.html', '.png'), width=1200, height=800)
        
        return fig
    
    def create_bubble_intensity_heatmap(self, 
                                       results_dict: Dict[str, Any],
                                       save_path: Optional[str] = None) -> go.Figure:
        """
        Create a heatmap showing bubble intensity across different metrics and systems.
        
        Args:
            results_dict: Dictionary mapping recommender names to SimulationResults
            save_path: Path to save the plot
            
        Returns:
            Plotly figure object
        """
        # Calculate bubble intensity metrics
        metrics_data = []
        metric_names = [
            'Diversity Loss (%)', 'Engagement Stability', 'Catalog Utilization',
            'User Satisfaction', 'Bubble Risk Score'
        ]
        
        for name, results in results_dict.items():
            # Calculate normalized metrics (higher = better, except bubble risk)
            diversity_loss = (results.content_diversity[0] - results.content_diversity[-1]) / results.content_diversity[0] * 100
            engagement_stability = 1.0 / (1.0 + np.var(results.engagement_rate))
            catalog_util = results.catalog_coverage[-1]
            user_satisfaction = results.avg_user_diversity[-1]
            bubble_risk = diversity_loss / 100.0 + (1.0 - engagement_stability)
            
            metrics_data.append([diversity_loss, engagement_stability, catalog_util, user_satisfaction, bubble_risk])
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=metrics_data,
            x=metric_names,
            y=list(results_dict.keys()),
            colorscale='RdYlGn_r',  # Red-Yellow-Green reversed (red = bad, green = good)
            hoverongaps=False,
            hovertemplate='<b>%{y}</b><br>' +
                         '%{x}: %{z:.3f}<br>' +
                         '<extra></extra>',
            colorbar=dict(
                title="Risk Level",
                titleside="right",
                tickmode="linear",
                tick0=0,
                dtick=0.2
            )
        ))
        
        fig.update_layout(
            title={
                'text': '<b>Filter Bubble Risk Assessment Matrix</b><br>' +
                       '<sub>Red = High Risk, Green = Low Risk</sub>',
                'x': 0.5,
                'font': {'size': 18, 'family': 'Arial, sans-serif'}
            },
            xaxis_title="Metrics",
            yaxis_title="Recommendation Systems",
            font=dict(family="Arial, sans-serif", size=12),
            height=400,
            width=800
        )
        
        if save_path:
            fig.write_html(save_path)
            self._save_plotly_as_png(fig, save_path.replace('.html', '.png'), width=800, height=400)
        
        return fig
    
    def create_user_journey_visualization(self, 
                                        interactions_df: pd.DataFrame,
                                        user_ids: List[int] = None,
                                        save_path: Optional[str] = None) -> go.Figure:
        """
        Create an animated visualization showing individual user journeys.
        
        Args:
            interactions_df: DataFrame with interaction data
            user_ids: Specific user IDs to visualize (if None, selects representative ones)
            save_path: Path to save the plot
            
        Returns:
            Plotly figure object
        """
        if user_ids is None:
            # Select diverse users for visualization
            user_diversity = interactions_df.groupby('user_id')['item_id'].nunique()
            # Get high, medium, and low diversity users
            user_ids = [
                user_diversity.idxmax(),  # Most diverse
                user_diversity.quantile(0.5, interpolation='nearest'),  # Medium
                user_diversity.idxmin()   # Least diverse (most bubbled)
            ]
        
        fig = go.Figure()
        
        colors = ['#E74C3C', '#F39C12', '#27AE60']  # Red, Orange, Green
        user_labels = ['Highly Bubbled User', 'Moderate User', 'Diverse User']
        
        for i, user_id in enumerate(user_ids):
            user_data = interactions_df[interactions_df['user_id'] == user_id].sort_values('timestep')
            
            # Calculate cumulative diversity
            cumulative_items = []
            unique_items = set()
            
            for _, row in user_data.iterrows():
                unique_items.add(row['item_id'])
                cumulative_items.append(len(unique_items))
            
            fig.add_trace(
                go.Scatter(
                    x=user_data['timestep'],
                    y=cumulative_items,
                    mode='lines+markers',
                    name=user_labels[i],
                    line=dict(color=colors[i], width=3),
                    marker=dict(size=8),
                    hovertemplate=f'<b>{user_labels[i]}</b><br>' +
                                 'Timestep: %{x}<br>' +
                                 'Unique Items Seen: %{y}<br>' +
                                 '<extra></extra>'
                )
            )
        
        fig.update_layout(
            title={
                'text': '<b>User Journey Comparison: Bubble Formation in Action</b><br>' +
                       '<sub>How Different Users Experience Content Diversity Over Time</sub>',
                'x': 0.5,
                'font': {'size': 18, 'family': 'Arial, sans-serif'}
            },
            xaxis_title="Simulation Timestep",
            yaxis_title="Cumulative Unique Items Seen",
            font=dict(family="Arial, sans-serif", size=12),
            plot_bgcolor='white',
            paper_bgcolor='white',
            height=600,
            width=900,
            legend=dict(
                orientation="v",
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="rgba(0,0,0,0.2)",
                borderwidth=1
            )
        )
        
        # Add annotations explaining the patterns
        fig.add_annotation(
            x=max(interactions_df['timestep']) * 0.7,
            y=max([len(interactions_df[interactions_df['user_id'] == uid]['item_id'].unique()) 
                  for uid in user_ids]) * 0.8,
            text="Diverse users continue<br>discovering new content",
            showarrow=True,
            arrowhead=2,
            arrowcolor=colors[2],
            font=dict(size=12, color=colors[2])
        )
        
        if save_path:
            fig.write_html(save_path)
            self._save_plotly_as_png(fig, save_path.replace('.html', '.png'), width=900, height=600)
        
        return fig
    
    def create_business_impact_dashboard(self, 
                                       results_dict: Dict[str, Any],
                                       save_path: Optional[str] = None) -> go.Figure:
        """
        Create a business-focused dashboard showing the trade-offs.
        
        Args:
            results_dict: Dictionary mapping recommender names to SimulationResults
            save_path: Path to save the plot
            
        Returns:
            Plotly figure object
        """
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=(
                'Short-term vs Long-term Engagement', 'Revenue Impact Over Time',
                'User Retention Proxy', 'Catalog ROI', 'Risk-Reward Analysis', 'Business Efficiency'
            ),
            specs=[[{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}]],
            vertical_spacing=0.15,
            horizontal_spacing=0.08
        )
        
        # Calculate business metrics for each system
        business_data = {}
        
        for name, results in results_dict.items():
            color = self.color_schemes.get(name, self.extended_colors[0])
            
            # Short-term vs long-term engagement
            short_term_engagement = np.mean(results.engagement_rate[:5])
            long_term_engagement = np.mean(results.engagement_rate[-5:])
            
            fig.add_trace(
                go.Bar(
                    x=[f"{name}<br>Short-term", f"{name}<br>Long-term"],
                    y=[short_term_engagement, long_term_engagement],
                    name=name,
                    marker_color=[color, color],
                    opacity=[1.0, 0.7],
                    showlegend=False
                ),
                row=1, col=1
            )
            
            # Revenue proxy over time (engagement * catalog coverage)
            revenue_proxy = [eng * cov for eng, cov in zip(results.engagement_rate, results.catalog_coverage)]
            
            fig.add_trace(
                go.Scatter(
                    x=results.timesteps,
                    y=revenue_proxy,
                    mode='lines',
                    name=name,
                    line=dict(color=color, width=3),
                    showlegend=False
                ),
                row=1, col=2
            )
            
            # User retention proxy (diversity maintenance)
            fig.add_trace(
                go.Scatter(
                    x=results.timesteps,
                    y=results.avg_user_diversity,
                    mode='lines',
                    name=name,
                    line=dict(color=color, width=3),
                    showlegend=False
                ),
                row=1, col=3
            )
            
            # Catalog ROI (coverage efficiency)
            catalog_roi = [cov / max(0.01, 1 - eng) for eng, cov in zip(results.engagement_rate, results.catalog_coverage)]
            
            fig.add_trace(
                go.Scatter(
                    x=results.timesteps,
                    y=catalog_roi,
                    mode='lines',
                    name=name,
                    line=dict(color=color, width=3),
                    showlegend=False
                ),
                row=2, col=1
            )
            
            # Store data for risk-reward analysis
            business_data[name] = {
                'avg_engagement': np.mean(results.engagement_rate),
                'diversity_risk': 1.0 - np.mean(results.content_diversity),
                'color': color
            }
        
        # Risk-Reward scatter plot
        for name, data in business_data.items():
            fig.add_trace(
                go.Scatter(
                    x=[data['diversity_risk']],
                    y=[data['avg_engagement']],
                    mode='markers+text',
                    text=[name],
                    textposition="top center",
                    marker=dict(
                        size=20,
                        color=data['color'],
                        line=dict(width=2, color='white')
                    ),
                    name=name,
                    showlegend=True
                ),
                row=2, col=2
            )
        
        # Business efficiency (engagement per unit diversity risk)
        efficiency_scores = [data['avg_engagement'] / max(0.01, data['diversity_risk']) 
                           for data in business_data.values()]
        
        fig.add_trace(
            go.Bar(
                x=list(business_data.keys()),
                y=efficiency_scores,
                marker_color=[data['color'] for data in business_data.values()],
                name='Efficiency',
                showlegend=False
            ),
            row=2, col=3
        )
        
        # Update layout
        fig.update_layout(
            title={
                'text': '<b>Business Impact Analysis: The True Cost of Filter Bubbles</b><br>' +
                       '<sub>Balancing Short-term Engagement with Long-term Business Health</sub>',
                'x': 0.5,
                'font': {'size': 18, 'family': 'Arial, sans-serif'}
            },
            font=dict(family="Arial, sans-serif", size=11),
            height=800,
            width=1400,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Update axis labels
        fig.update_xaxes(title_text="Engagement Period", row=1, col=1)
        fig.update_xaxes(title_text="Timestep", row=1, col=2)
        fig.update_xaxes(title_text="Timestep", row=1, col=3)
        fig.update_xaxes(title_text="Timestep", row=2, col=1)
        fig.update_xaxes(title_text="Diversity Risk", row=2, col=2)
        fig.update_xaxes(title_text="Recommendation System", row=2, col=3)
        
        fig.update_yaxes(title_text="Engagement Rate", row=1, col=1)
        fig.update_yaxes(title_text="Revenue Proxy", row=1, col=2)
        fig.update_yaxes(title_text="User Retention", row=1, col=3)
        fig.update_yaxes(title_text="Catalog ROI", row=2, col=1)
        fig.update_yaxes(title_text="Avg Engagement", row=2, col=2)
        fig.update_yaxes(title_text="Efficiency Score", row=2, col=3)
        
        if save_path:
            fig.write_html(save_path)
            self._save_plotly_as_png(fig, save_path.replace('.html', '.png'), width=1400, height=800)
        
        return fig
    
    def create_exploration_vs_exploitation_animation(self, 
                                                   rl_metrics: Dict[str, List[float]],
                                                   save_path: Optional[str] = None) -> go.Figure:
        """
        Create an animated plot showing exploration vs exploitation balance in RL.
        
        Args:
            rl_metrics: Dictionary containing RL-specific metrics over time
            save_path: Path to save the plot
            
        Returns:
            Plotly figure object
        """
        timesteps = list(range(len(rl_metrics.get('epsilon', []))))
        
        fig = go.Figure()
        
        # Add epsilon decay line
        if 'epsilon' in rl_metrics:
            fig.add_trace(
                go.Scatter(
                    x=timesteps,
                    y=rl_metrics['epsilon'],
                    mode='lines+markers',
                    name='Exploration Rate (ε)',
                    line=dict(color='#E74C3C', width=3),
                    marker=dict(size=8),
                    yaxis='y'
                )
            )
        
        # Add diversity maintenance line
        if 'avg_reward' in rl_metrics:
            # Normalize rewards to 0-1 scale for comparison
            normalized_rewards = np.array(rl_metrics['avg_reward'])
            normalized_rewards = (normalized_rewards - normalized_rewards.min()) / (normalized_rewards.max() - normalized_rewards.min())
            
            fig.add_trace(
                go.Scatter(
                    x=timesteps,
                    y=normalized_rewards,
                    mode='lines+markers',
                    name='Normalized Reward',
                    line=dict(color='#27AE60', width=3),
                    marker=dict(size=8),
                    yaxis='y'
                )
            )
        
        # Add phase annotations
        phases = [
            {"start": 0, "end": len(timesteps)//3, "color": "rgba(255,0,0,0.1)", "label": "High Exploration"},
            {"start": len(timesteps)//3, "end": 2*len(timesteps)//3, "color": "rgba(255,165,0,0.1)", "label": "Balanced"},
            {"start": 2*len(timesteps)//3, "end": len(timesteps), "color": "rgba(0,128,0,0.1)", "label": "Exploitation Focus"}
        ]
        
        for phase in phases:
            fig.add_vrect(
                x0=phase["start"], x1=phase["end"],
                fillcolor=phase["color"],
                layer="below",
                line_width=0,
                annotation_text=phase["label"],
                annotation_position="top left"
            )
        
        fig.update_layout(
            title={
                'text': '<b>RL Learning Dynamics: Exploration vs Exploitation Balance</b><br>' +
                       '<sub>How Reinforcement Learning Adapts Over Time to Prevent Bubbles</sub>',
                'x': 0.5,
                'font': {'size': 18, 'family': 'Arial, sans-serif'}
            },
            xaxis_title="Training Timestep",
            yaxis_title="Value",
            font=dict(family="Arial, sans-serif", size=12),
            plot_bgcolor='white',
            paper_bgcolor='white',
            height=600,
            width=1000
        )
        
        if save_path:
            fig.write_html(save_path)
            self._save_plotly_as_png(fig, save_path.replace('.html', '.png'), width=1000, height=600)
        
        return fig
    
    def create_statistical_significance_plot(self, 
                                           comparative_metrics: Dict[str, Any],
                                           save_path: Optional[str] = None) -> go.Figure:
        """
        Create a plot showing statistical significance of differences between systems.
        
        Args:
            comparative_metrics: Dictionary containing comparative analysis results
            save_path: Path to save the plot
            
        Returns:
            Plotly figure object
        """
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Diversity Differences', 'Engagement Differences'),
            horizontal_spacing=0.15
        )
        
        # Extract statistical test results
        if 'statistical_tests' in comparative_metrics:
            diversity_tests = comparative_metrics['statistical_tests'].get('diversity_tests', {})
            engagement_tests = comparative_metrics['statistical_tests'].get('engagement_tests', {})
            
            # Plot diversity test results
            comparisons = list(diversity_tests.keys())
            p_values_div = [test['p_value'] for test in diversity_tests.values()]
            significance_div = [test['significant'] for test in diversity_tests.values()]
            
            colors_div = ['#27AE60' if sig else '#E74C3C' for sig in significance_div]
            
            fig.add_trace(
                go.Bar(
                    x=comparisons,
                    y=[-np.log10(p) for p in p_values_div],
                    marker_color=colors_div,
                    name='Diversity Tests',
                    showlegend=False,
                    hovertemplate='<b>%{x}</b><br>' +
                                 '-log10(p-value): %{y:.2f}<br>' +
                                 'Significant: %{customdata}<br>' +
                                 '<extra></extra>',
                    customdata=significance_div
                ),
                row=1, col=1
            )
            
            # Plot engagement test results
            if engagement_tests:
                p_values_eng = [test['p_value'] for test in engagement_tests.values()]
                significance_eng = [test['significant'] for test in engagement_tests.values()]
                colors_eng = ['#27AE60' if sig else '#E74C3C' for sig in significance_eng]
                
                fig.add_trace(
                    go.Bar(
                        x=comparisons,
                        y=[-np.log10(p) for p in p_values_eng],
                        marker_color=colors_eng,
                        name='Engagement Tests',
                        showlegend=False,
                        hovertemplate='<b>%{x}</b><br>' +
                                     '-log10(p-value): %{y:.2f}<br>' +
                                     'Significant: %{customdata}<br>' +
                                     '<extra></extra>',
                        customdata=significance_eng
                    ),
                    row=1, col=2
                )
        
        # Add significance threshold line
        fig.add_hline(y=-np.log10(0.05), line_dash="dash", line_color="red", 
                     annotation_text="p=0.05 threshold", row=1, col=1)
        fig.add_hline(y=-np.log10(0.05), line_dash="dash", line_color="red", 
                     annotation_text="p=0.05 threshold", row=1, col=2)
        
        fig.update_layout(
            title={
                'text': '<b>Statistical Significance of Recommender Differences</b><br>' +
                       '<sub>Green = Significant Difference, Red = No Significant Difference</sub>',
                'x': 0.5,
                'font': {'size': 16, 'family': 'Arial, sans-serif'}
            },
            font=dict(family="Arial, sans-serif", size=12),
            height=500,
            width=1000
        )
        
        fig.update_yaxes(title_text="-log10(p-value)", row=1, col=1)
        fig.update_yaxes(title_text="-log10(p-value)", row=1, col=2)
        fig.update_xaxes(title_text="System Comparison", row=1, col=1)
        fig.update_xaxes(title_text="System Comparison", row=1, col=2)
        
        if save_path:
            fig.write_html(save_path)
            self._save_plotly_as_png(fig, save_path.replace('.html', '.png'), width=1000, height=500)
        
        return fig
    
    def create_comprehensive_report_figure(self, 
                                         results_dict: Dict[str, Any],
                                         interactions_dict: Dict[str, pd.DataFrame],
                                         comparative_metrics: Dict[str, Any],
                                         save_path: Optional[str] = None) -> go.Figure:
        """
        Create a comprehensive figure suitable for publication/blog post.
        
        This is the ultimate visualization that tells the complete story.
        
        Args:
            results_dict: Dictionary mapping recommender names to SimulationResults
            interactions_dict: Dictionary mapping recommender names to interaction DataFrames
            comparative_metrics: Dictionary containing comparative analysis results
            save_path: Path to save the plot
            
        Returns:
            Plotly figure object
        """
        fig = make_subplots(
            rows=3, cols=3,
            subplot_titles=(
                'Filter Bubble Formation', 'Business Impact: Engagement vs Diversity',
                'Content Distribution Inequality',
                'User Journey Divergence', 'Long-term vs Short-term Performance',
                'Exploration Strategy Effectiveness',
                'Statistical Significance', 'ROI Analysis', 'Recommendation Matrix'
            ),
            specs=[
                [{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}, {"type": "table"}]
            ],
            vertical_spacing=0.08,
            horizontal_spacing=0.06
        )
        
        # 1. Filter Bubble Formation (Main Story)
        for name, results in results_dict.items():
            color = self.color_schemes.get(name, self.extended_colors[0])
            
            fig.add_trace(
                go.Scatter(
                    x=results.timesteps,
                    y=results.content_diversity,
                    mode='lines+markers',
                    name=name,
                    line=dict(color=color, width=3),
                    marker=dict(size=6),
                    legendgroup=name
                ),
                row=1, col=1
            )
        
        # 2. Engagement vs Diversity Scatter
        for name, results in results_dict.items():
            color = self.color_schemes.get(name, self.extended_colors[0])
            avg_engagement = np.mean(results.engagement_rate)
            avg_diversity = np.mean(results.content_diversity)
            
            fig.add_trace(
                go.Scatter(
                    x=[avg_diversity],
                    y=[avg_engagement],
                    mode='markers+text',
                    text=[name],
                    textposition="top center",
                    marker=dict(size=20, color=color, line=dict(width=2, color='white')),
                    name=name,
                    showlegend=False,
                    legendgroup=name
                ),
                row=1, col=2
            )
        
        # 3. Content Distribution (Gini-like coefficient)
        gini_scores = {}
        for name, interactions_df in interactions_dict.items():
            item_counts = interactions_df['item_id'].value_counts().values
            if len(item_counts) > 0:
                n = len(item_counts)
                sorted_counts = np.sort(item_counts)
                cumsum = np.cumsum(sorted_counts)
                gini = (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n
                gini_scores[name] = max(0.0, gini)
            else:
                gini_scores[name] = 0.0
        
        fig.add_trace(
            go.Bar(
                x=list(gini_scores.keys()),
                y=list(gini_scores.values()),
                marker_color=[self.color_schemes.get(name, '#95A5A6') for name in gini_scores.keys()],
                name='Content Inequality',
                showlegend=False
            ),
            row=1, col=3
        )
        
        # 4. User Journey Divergence
        for name, interactions_df in interactions_dict.items():
            color = self.color_schemes.get(name, self.extended_colors[0])
            
            # Calculate user diversity variance over time
            diversity_variance = []
            for timestep in sorted(interactions_df['timestep'].unique()):
                timestep_data = interactions_df[interactions_df['timestep'] == timestep]
                user_diversities = timestep_data.groupby('user_id')['item_id'].nunique()
                diversity_variance.append(user_diversities.var())
            
            timesteps = sorted(interactions_df['timestep'].unique())
            
            fig.add_trace(
                go.Scatter(
                    x=timesteps,
                    y=diversity_variance,
                    mode='lines',
                    name=name,
                    line=dict(color=color, width=2),
                    showlegend=False,
                    legendgroup=name
                ),
                row=2, col=1
            )
        
        # 5. Long-term vs Short-term Performance
        performance_data = []
        system_names = []
        
        for name, results in results_dict.items():
            short_term = np.mean(results.engagement_rate[:len(results.engagement_rate)//3])
            long_term = np.mean(results.engagement_rate[-len(results.engagement_rate)//3:])
            
            performance_data.extend([short_term, long_term])
            system_names.extend([f"{name}<br>(Short)", f"{name}<br>(Long)"])
        
        colors_perf = []
        for name in results_dict.keys():
            color = self.color_schemes.get(name, self.extended_colors[0])
            colors_perf.extend([color, color])
        
        fig.add_trace(
            go.Bar(
                x=system_names,
                y=performance_data,
                marker_color=colors_perf,
                opacity=[1.0, 0.7] * len(results_dict),
                name='Performance',
                showlegend=False
            ),
            row=2, col=2
        )
        
        # 6. Exploration Strategy (if RL metrics available)
        rl_systems = [name for name in results_dict.keys() if 'RL' in name or 'rl' in name.lower()]
        if rl_systems and results_dict[rl_systems[0]].recommender_metrics:
            rl_name = rl_systems[0]
            rl_metrics = results_dict[rl_name].recommender_metrics
            
            if 'epsilon' in rl_metrics:
                fig.add_trace(
                    go.Scatter(
                        x=results_dict[rl_name].timesteps,
                        y=rl_metrics['epsilon'],
                        mode='lines+markers',
                        name='Exploration Rate',
                        line=dict(color='#E74C3C', width=2),
                        showlegend=False
                    ),
                    row=2, col=3
                )
        
        # 7. Statistical Significance (simplified)
        if 'statistical_tests' in comparative_metrics:
            diversity_tests = comparative_metrics['statistical_tests'].get('diversity_tests', {})
            if diversity_tests:
                comparisons = list(diversity_tests.keys())
                p_values = [test['p_value'] for test in diversity_tests.values()]
                significance = [test['significant'] for test in diversity_tests.values()]
                
                colors_stat = ['#27AE60' if sig else '#E74C3C' for sig in significance]
                
                fig.add_trace(
                    go.Bar(
                        x=comparisons,
                        y=[-np.log10(p) for p in p_values],
                        marker_color=colors_stat,
                        name='Significance',
                        showlegend=False
                    ),
                    row=3, col=1
                )
        
        # 8. ROI Analysis
        roi_data = {}
        for name, results in results_dict.items():
            engagement_benefit = np.mean(results.engagement_rate)
            diversity_cost = 1.0 - np.mean(results.content_diversity)
            roi = engagement_benefit / max(0.01, diversity_cost)
            roi_data[name] = roi
        
        fig.add_trace(
            go.Bar(
                x=list(roi_data.keys()),
                y=list(roi_data.values()),
                marker_color=[self.color_schemes.get(name, '#95A5A6') for name in roi_data.keys()],
                name='ROI',
                showlegend=False
            ),
            row=3, col=2
        )
        
        # 9. Recommendation Matrix (Table)
        recommendations = []
        scores = []
        
        for name, results in results_dict.items():
            diversity_score = np.mean(results.content_diversity)
            engagement_score = np.mean(results.engagement_rate)
            balance_score = (diversity_score + engagement_score) / 2
            
            if balance_score > 0.7:
                recommendation = "✅ Recommended"
            elif balance_score > 0.5:
                recommendation = "⚠️ Conditional"
            else:
                recommendation = "❌ Not Recommended"
            
            recommendations.append(recommendation)
            scores.append(f"{balance_score:.3f}")
        
        fig.add_trace(
            go.Table(
                header=dict(
                    values=['System', 'Score', 'Recommendation'],
                    fill_color='lightgray',
                    font=dict(size=12, color='black')
                ),
                cells=dict(
                    values=[list(results_dict.keys()), scores, recommendations],
                    fill_color='white',
                    font=dict(size=11)
                )
            ),
            row=3, col=3
        )
        
        # Update layout
        fig.update_layout(
            title={
                'text': '<b>Comprehensive Filter Bubble Analysis Report</b><br>' +
                       '<sub>Complete Performance Comparison Across All Metrics</sub>',
                'x': 0.5,
                'font': {'size': 20, 'family': 'Arial, sans-serif'}
            },
            font=dict(family="Arial, sans-serif", size=10),
            height=1200,
            width=1600,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5
            )
        )
        
        # Update axes labels
        axes_labels = [
            ("Timestep", "Content Diversity", 1, 1),
            ("Average Diversity", "Average Engagement", 1, 2),
            ("System", "Gini Coefficient", 1, 3),
            ("Timestep", "User Diversity Variance", 2, 1),
            ("System & Period", "Engagement Rate", 2, 2),
            ("Timestep", "Exploration Rate", 2, 3),
            ("System Comparison", "-log10(p-value)", 3, 1),
            ("System", "ROI Score", 3, 2)
        ]
        
        for xlabel, ylabel, row, col in axes_labels:
            fig.update_xaxes(title_text=xlabel, row=row, col=col, title_font=dict(size=11))
            fig.update_yaxes(title_text=ylabel, row=row, col=col, title_font=dict(size=11))
        
        if save_path:
            fig.write_html(save_path)
            self._save_plotly_as_png(fig, save_path.replace('.html', '.png'), width=1600, height=1200)
        
        return fig


class InteractiveDashboard:
    """
    Interactive dashboard for exploring filter bubble dynamics.
    """
    
    def __init__(self, visualization_suite: BubbleVisualizationSuite):
        """
        Initialize the interactive dashboard.
        
        Args:
            visualization_suite: The visualization suite to use
        """
        self.viz_suite = visualization_suite
    
    def create_interactive_exploration_dashboard(self, 
                                               results_dict: Dict[str, Any],
                                               interactions_dict: Dict[str, pd.DataFrame],
                                               save_path: Optional[str] = None) -> go.Figure:
        """
        Create an interactive dashboard for exploring the results.
        
        Args:
            results_dict: Dictionary mapping recommender names to SimulationResults
            interactions_dict: Dictionary mapping recommender names to interaction DataFrames
            save_path: Path to save the plot
            
        Returns:
            Plotly figure object
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Metric Comparison Over Time', 'User Engagement Distribution',
                'Content Popularity Analysis', 'System Performance Radar'
            ),
            specs=[
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"type": "polar"}]
            ]
        )
        
        # Create dropdown for metric selection
        _metrics_options = [
            'content_diversity', 'engagement_rate', 'catalog_coverage', 'avg_user_diversity'
        ]
        
        # Initial plot with content diversity
        for name, results in results_dict.items():
            color = self.viz_suite.color_schemes.get(name, self.viz_suite.extended_colors[0])
            
            fig.add_trace(
                go.Scatter(
                    x=results.timesteps,
                    y=results.content_diversity,
                    mode='lines+markers',
                    name=name,
                    line=dict(color=color, width=3),
                    visible=True
                ),
                row=1, col=1
            )
        
        # User engagement distribution
        for name, interactions_df in interactions_dict.items():
            color = self.viz_suite.color_schemes.get(name, self.viz_suite.extended_colors[0])
            
            user_engagement = interactions_df.groupby('user_id')['engaged'].mean()
            
            fig.add_trace(
                go.Histogram(
                    x=user_engagement,
                    name=f'{name} Users',
                    opacity=0.7,
                    marker_color=color,
                    nbinsx=20
                ),
                row=1, col=2
            )
        
        # Content popularity analysis
        for name, interactions_df in interactions_dict.items():
            color = self.viz_suite.color_schemes.get(name, self.viz_suite.extended_colors[0])
            
            item_popularity = interactions_df['item_popularity'].values
            item_interactions = interactions_df.groupby('item_id').size().values
            
            fig.add_trace(
                go.Scatter(
                    x=item_popularity,
                    y=item_interactions,
                    mode='markers',
                    name=f'{name} Content',
                    marker=dict(
                        color=color,
                        size=8,
                        opacity=0.6
                    )
                ),
                row=2, col=1
            )
        
        # Performance radar chart
        categories = ['Diversity', 'Engagement', 'Coverage', 'User Satisfaction', 'Efficiency']
        
        for name, results in results_dict.items():
            color = self.viz_suite.color_schemes.get(name, self.viz_suite.extended_colors[0])
            
            # Normalize metrics to 0-1 scale
            diversity_score = np.mean(results.content_diversity)
            engagement_score = np.mean(results.engagement_rate)
            coverage_score = np.mean(results.catalog_coverage)
            satisfaction_score = np.mean(results.avg_user_diversity)
            efficiency_score = (diversity_score + engagement_score) / 2
            
            values = [diversity_score, engagement_score, coverage_score, satisfaction_score, efficiency_score]
            
            fig.add_trace(
                go.Scatterpolar(
                    r=values,
                    theta=categories,
                    fill='toself',
                    name=name,
                    line_color=color,
                    fillcolor=color,
                    opacity=0.3
                ),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            title={
                'text': '<b>Interactive Filter Bubble Exploration Dashboard</b>',
                'x': 0.5,
                'font': {'size': 18, 'family': 'Arial, sans-serif'}
            },
            height=800,
            width=1200
        )
        
        # Update polar plot
        fig.update_polars(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        )
        
        # Update axes
        fig.update_xaxes(title_text="Timestep", row=1, col=1)
        fig.update_yaxes(title_text="Content Diversity", row=1, col=1)
        fig.update_xaxes(title_text="Engagement Rate", row=1, col=2)
        fig.update_yaxes(title_text="Number of Users", row=1, col=2)
        fig.update_xaxes(title_text="Item Popularity", row=2, col=1)
        fig.update_yaxes(title_text="Number of Interactions", row=2, col=1)
        
        if save_path:
            fig.write_html(save_path)
            self._save_plotly_as_png(fig, save_path.replace('.html', '.png'), width=1200, height=800)
        
        return fig


# Utility functions for creating publication-ready plots
def save_publication_figures(viz_suite: BubbleVisualizationSuite,
                           results_dict: Dict[str, Any],
                           interactions_dict: Dict[str, pd.DataFrame],
                           comparative_metrics: Dict[str, Any],
                           output_dir: str = "./figures/") -> Dict[str, str]:
    """
    Generate and save all publication-ready figures.
    
    Args:
        viz_suite: The visualization suite
        results_dict: Dictionary mapping recommender names to SimulationResults
        interactions_dict: Dictionary mapping recommender names to interaction DataFrames
        comparative_metrics: Dictionary containing comparative analysis results
        output_dir: Directory to save figures
        
    Returns:
        Dictionary mapping figure names to file paths
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    saved_figures = {}
    
    # 1. Main story figure
    _main_story = viz_suite.create_bubble_formation_story(
        results_dict,
        save_path=os.path.join(output_dir, "bubble_formation_story.html")
    )
    saved_figures["main_story"] = os.path.join(output_dir, "bubble_formation_story.png")
    
    # 2. Business impact dashboard
    _business_impact = viz_suite.create_business_impact_dashboard(
        results_dict,
        save_path=os.path.join(output_dir, "business_impact_dashboard.html")
    )
    saved_figures["business_impact"] = os.path.join(output_dir, "business_impact_dashboard.png")
    
    # 3. Comprehensive report
    _comprehensive = viz_suite.create_comprehensive_report_figure(
        results_dict,
        interactions_dict,
        comparative_metrics,
        save_path=os.path.join(output_dir, "comprehensive_report.html")
    )
    saved_figures["comprehensive"] = os.path.join(output_dir, "comprehensive_report.png")
    
    # 4. User journey visualization
    if interactions_dict:
        sample_interactions = list(interactions_dict.values())[0]
        _user_journey = viz_suite.create_user_journey_visualization(
            sample_interactions,
            save_path=os.path.join(output_dir, "user_journey.html")
        )
        saved_figures["user_journey"] = os.path.join(output_dir, "user_journey.png")
    
    # 5. Statistical significance
    if comparative_metrics:
        _significance = viz_suite.create_statistical_significance_plot(
            comparative_metrics,
            save_path=os.path.join(output_dir, "statistical_significance.html")
        )
        saved_figures["significance"] = os.path.join(output_dir, "statistical_significance.png")
    
    return saved_figures