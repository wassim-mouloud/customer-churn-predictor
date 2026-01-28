"""Plotly visualization functions with dark theme for churn analysis."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from .config import COLORS, CHART_COLORS, RISK_COLORS


def get_layout_template() -> dict:
    """Return dark theme layout template for Plotly charts."""
    return {
        'paper_bgcolor': COLORS['background'],
        'plot_bgcolor': COLORS['paper'],
        'font': {'color': COLORS['text'], 'family': 'Arial, sans-serif'},
        'title_font': {'size': 20, 'color': COLORS['primary']},
        'xaxis': {
            'gridcolor': COLORS['grid'],
            'linecolor': COLORS['grid'],
            'zerolinecolor': COLORS['grid']
        },
        'yaxis': {
            'gridcolor': COLORS['grid'],
            'linecolor': COLORS['grid'],
            'zerolinecolor': COLORS['grid']
        },
        'legend_bgcolor': 'rgba(0,0,0,0)',
        'margin': {'t': 60, 'b': 40, 'l': 40, 'r': 40}
    }


def plot_churn_distribution(df: pd.DataFrame) -> go.Figure:
    """Create donut chart showing churn distribution."""
    churn_counts = df['churn'].value_counts()
    labels = ['Retained', 'Churned']
    values = [churn_counts.get(0, 0), churn_counts.get(1, 0)]

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.6,
        marker=dict(colors=[COLORS['success'], COLORS['danger']]),
        textinfo='label+percent',
        textfont=dict(size=14, color=COLORS['text']),
        hovertemplate='%{label}<br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
    )])

    total = sum(values)
    churn_rate = values[1] / total * 100

    fig.update_layout(
        **get_layout_template(),
        title='Customer Churn Distribution',
        showlegend=True,
        annotations=[{
            'text': f'{churn_rate:.1f}%<br>Churn Rate',
            'x': 0.5, 'y': 0.5,
            'font': {'size': 20, 'color': COLORS['text']},
            'showarrow': False
        }]
    )

    return fig


def plot_tenure_distribution(df: pd.DataFrame) -> go.Figure:
    """Create histogram showing tenure distribution by churn status."""
    fig = go.Figure()

    for churn_val, name, color in [(0, 'Retained', COLORS['success']), (1, 'Churned', COLORS['danger'])]:
        subset = df[df['churn'] == churn_val]['tenure']
        fig.add_trace(go.Histogram(
            x=subset,
            name=name,
            marker_color=color,
            opacity=0.7,
            nbinsx=20
        ))

    fig.update_layout(
        **get_layout_template(),
        title='Customer Tenure Distribution',
        xaxis_title='Tenure (Months)',
        yaxis_title='Number of Customers',
        barmode='overlay',
        legend=dict(x=0.8, y=0.95)
    )

    return fig


def plot_charges_scatter(df: pd.DataFrame) -> go.Figure:
    """Create scatter plot of monthly vs total charges colored by churn."""
    fig = go.Figure()

    for churn_val, name, color in [(0, 'Retained', COLORS['success']), (1, 'Churned', COLORS['danger'])]:
        subset = df[df['churn'] == churn_val]
        fig.add_trace(go.Scatter(
            x=subset['monthly_charges'],
            y=subset['total_charges'],
            mode='markers',
            name=name,
            marker=dict(color=color, size=5, opacity=0.5),
            hovertemplate=(
                'Monthly: $%{x:.2f}<br>'
                'Total: $%{y:.2f}<br>'
                '<extra></extra>'
            )
        ))

    fig.update_layout(
        **get_layout_template(),
        title='Monthly vs Total Charges',
        xaxis_title='Monthly Charges ($)',
        yaxis_title='Total Charges ($)',
        legend=dict(x=0.02, y=0.98)
    )

    return fig


def plot_churn_by_contract(df: pd.DataFrame) -> go.Figure:
    """Create grouped bar chart showing churn rate by contract type."""
    from .config import CONTRACT_LABELS

    grouped = df.groupby(['contract_type', 'churn']).size().unstack(fill_value=0)
    grouped['churn_rate'] = grouped[1] / (grouped[0] + grouped[1]) * 100

    contracts = [CONTRACT_LABELS[i] for i in grouped.index]
    churn_rates = grouped['churn_rate'].values
    total_customers = (grouped[0] + grouped[1]).values

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=contracts,
        y=churn_rates,
        marker_color=[COLORS['danger'] if rate > 30 else COLORS['warning'] if rate > 15 else COLORS['success']
                      for rate in churn_rates],
        text=[f'{rate:.1f}%' for rate in churn_rates],
        textposition='outside',
        hovertemplate='Contract: %{x}<br>Churn Rate: %{y:.1f}%<br>Total Customers: %{customdata}<extra></extra>',
        customdata=total_customers
    ))

    layout = get_layout_template()
    layout['yaxis'] = {**layout['yaxis'], 'range': [0, max(churn_rates) * 1.2]}
    fig.update_layout(
        **layout,
        title='Churn Rate by Contract Type',
        xaxis_title='Contract Type',
        yaxis_title='Churn Rate (%)',
        showlegend=False
    )

    return fig


def plot_churn_by_service(df: pd.DataFrame) -> go.Figure:
    """Create grouped bar chart showing churn rate by internet service."""
    from .config import INTERNET_LABELS

    grouped = df.groupby(['internet_service', 'churn']).size().unstack(fill_value=0)
    grouped['churn_rate'] = grouped[1] / (grouped[0] + grouped[1]) * 100

    services = [INTERNET_LABELS[i] for i in grouped.index]
    churn_rates = grouped['churn_rate'].values

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=services,
        y=churn_rates,
        marker_color=CHART_COLORS[:len(services)],
        text=[f'{rate:.1f}%' for rate in churn_rates],
        textposition='outside'
    ))

    layout = get_layout_template()
    layout['yaxis'] = {**layout['yaxis'], 'range': [0, max(churn_rates) * 1.2]}
    fig.update_layout(
        **layout,
        title='Churn Rate by Internet Service',
        xaxis_title='Internet Service',
        yaxis_title='Churn Rate (%)',
        showlegend=False
    )

    return fig


def plot_correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    """Create interactive correlation heatmap."""
    from .config import FEATURE_COLUMNS

    numeric_cols = FEATURE_COLUMNS + ['churn']
    corr_matrix = df[numeric_cols].corr()

    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu_r',
        zmid=0,
        text=np.round(corr_matrix.values, 2),
        texttemplate='%{text}',
        textfont={'size': 10},
        hovertemplate='%{x} vs %{y}<br>Correlation: %{z:.3f}<extra></extra>'
    ))

    layout = get_layout_template()
    layout['xaxis'] = {**layout['xaxis'], 'tickangle': 45}
    fig.update_layout(
        **layout,
        title='Feature Correlation Matrix',
        height=600
    )

    return fig


def plot_confusion_matrix(cm: np.ndarray) -> go.Figure:
    """Create interactive confusion matrix visualization."""
    labels = ['Retained', 'Churned']

    # Calculate percentages
    cm_pct = cm / cm.sum() * 100

    text = [[f'{cm[i][j]}<br>({cm_pct[i][j]:.1f}%)' for j in range(2)] for i in range(2)]

    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=['Predicted: ' + l for l in labels],
        y=['Actual: ' + l for l in labels],
        colorscale=[[0, COLORS['paper']], [1, COLORS['primary']]],
        text=text,
        texttemplate='%{text}',
        textfont={'size': 16},
        hovertemplate='%{y}<br>%{x}<br>Count: %{z}<extra></extra>',
        showscale=False
    ))

    fig.update_layout(
        **get_layout_template(),
        title='Confusion Matrix',
        xaxis_title='Predicted',
        yaxis_title='Actual',
        height=400
    )

    return fig


def plot_roc_curve(fpr: np.ndarray, tpr: np.ndarray, auc: float) -> go.Figure:
    """Create ROC curve with AUC annotation."""
    fig = go.Figure()

    # ROC curve
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr,
        mode='lines',
        name=f'ROC Curve (AUC = {auc:.3f})',
        line=dict(color=COLORS['primary'], width=3),
        fill='tozeroy',
        fillcolor=f'rgba(0, 212, 255, 0.2)'
    ))

    # Diagonal line
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Random Classifier',
        line=dict(color=COLORS['text_secondary'], width=2, dash='dash')
    ))

    layout = get_layout_template()
    layout['xaxis'] = {**layout['xaxis'], 'range': [0, 1]}
    layout['yaxis'] = {**layout['yaxis'], 'range': [0, 1.05]}
    fig.update_layout(
        **layout,
        title=f'ROC Curve (AUC = {auc:.3f})',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        legend=dict(x=0.5, y=0.1)
    )

    return fig


def plot_precision_recall_curve(precision: np.ndarray, recall: np.ndarray) -> go.Figure:
    """Create Precision-Recall curve."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=recall, y=precision,
        mode='lines',
        name='PR Curve',
        line=dict(color=COLORS['secondary'], width=3),
        fill='tozeroy',
        fillcolor=f'rgba(187, 134, 252, 0.2)'
    ))

    layout = get_layout_template()
    layout['xaxis'] = {**layout['xaxis'], 'range': [0, 1]}
    layout['yaxis'] = {**layout['yaxis'], 'range': [0, 1.05]}
    fig.update_layout(
        **layout,
        title='Precision-Recall Curve',
        xaxis_title='Recall',
        yaxis_title='Precision'
    )

    return fig


def plot_feature_importance(importance_dict: dict) -> go.Figure:
    """Create horizontal bar chart of feature importances."""
    # Sort by importance
    sorted_items = sorted(importance_dict.items(), key=lambda x: x[1])
    features = [item[0].replace('_', ' ').title() for item in sorted_items]
    importances = [item[1] for item in sorted_items]

    # Color based on importance
    colors = [COLORS['primary'] if imp > np.mean(importances) else COLORS['text_secondary']
              for imp in importances]

    fig = go.Figure(data=go.Bar(
        x=importances,
        y=features,
        orientation='h',
        marker_color=colors,
        text=[f'{imp:.3f}' for imp in importances],
        textposition='outside',
        hovertemplate='%{y}<br>Importance: %{x:.4f}<extra></extra>'
    ))

    layout = get_layout_template()
    layout['margin'] = {**layout['margin'], 'l': 150}
    fig.update_layout(
        **layout,
        title='Feature Importance (Random Forest)',
        xaxis_title='Importance Score',
        yaxis_title='',
        height=500
    )

    return fig


def plot_churn_gauge(probability: float) -> go.Figure:
    """Create gauge chart showing churn probability."""
    # Determine color based on probability
    if probability < 0.3:
        color = RISK_COLORS['low']
    elif probability < 0.6:
        color = RISK_COLORS['medium']
    else:
        color = RISK_COLORS['high']

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        number={'suffix': '%', 'font': {'size': 50, 'color': COLORS['text']}},
        gauge={
            'axis': {
                'range': [0, 100],
                'tickcolor': COLORS['text'],
                'tickfont': {'color': COLORS['text']}
            },
            'bar': {'color': color, 'thickness': 0.8},
            'bgcolor': COLORS['paper'],
            'borderwidth': 2,
            'bordercolor': COLORS['grid'],
            'steps': [
                {'range': [0, 30], 'color': 'rgba(0, 230, 118, 0.3)'},
                {'range': [30, 60], 'color': 'rgba(255, 217, 61, 0.3)'},
                {'range': [60, 100], 'color': 'rgba(255, 68, 68, 0.3)'}
            ],
            'threshold': {
                'line': {'color': COLORS['text'], 'width': 4},
                'thickness': 0.8,
                'value': probability * 100
            }
        }
    ))

    fig.update_layout(
        paper_bgcolor=COLORS['background'],
        font={'color': COLORS['text']},
        height=300,
        margin=dict(t=30, b=30, l=30, r=30)
    )

    return fig


def plot_monthly_charges_distribution(df: pd.DataFrame) -> go.Figure:
    """Create box plot of monthly charges by churn status."""
    fig = go.Figure()

    for churn_val, name, color in [(0, 'Retained', COLORS['success']), (1, 'Churned', COLORS['danger'])]:
        subset = df[df['churn'] == churn_val]['monthly_charges']
        fig.add_trace(go.Box(
            y=subset,
            name=name,
            marker_color=color,
            boxmean=True
        ))

    fig.update_layout(
        **get_layout_template(),
        title='Monthly Charges by Churn Status',
        yaxis_title='Monthly Charges ($)',
        showlegend=True
    )

    return fig
