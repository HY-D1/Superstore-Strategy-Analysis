"""
Superstore Strategy Analysis - Streamlit Dashboard
Interactive display of all findings, charts, and outputs
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Page config
st.set_page_config(
    page_title="Superstore Strategy Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #333;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
    }
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
    }
    .insight-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)


def load_data(file_path: str) -> pd.DataFrame | None:
    """Load CSV or Excel data with error handling."""
    path = Path(file_path)
    if not path.exists():
        return None
    try:
        if path.suffix == '.csv':
            df = pd.read_csv(path)
            return df if not df.empty else None
        elif path.suffix in ['.xlsx', '.xls']:
            df = pd.read_excel(path, sheet_name='Master')
            return df if not df.empty else None
        return None
    except Exception:
        return None


def load_first_available(*paths: str) -> pd.DataFrame | None:
    """Load the first available data file from a list of paths."""
    for path in paths:
        df = load_data(path)
        if df is not None:
            return df
    return None


def load_markdown(file_path: str) -> str:
    """Load markdown file content."""
    path = Path(file_path)
    if path.exists():
        return path.read_text(encoding='utf-8')
    return "*File not found*"


def format_currency(value: float) -> str:
    """Format value as currency."""
    return f"${value:,.0f}"


def format_percentage(value: float) -> str:
    """Format value as percentage."""
    return f"{value*100:.1f}%"


# =============================================================================
# Sidebar Navigation
# =============================================================================
st.sidebar.title("📊 Navigation")
page = st.sidebar.radio(
    "Select a section:",
    ["🏠 Executive Summary", "📈 Sales Insights", "🔮 Forecast", "👥 RFM Segmentation", "📦 Data Explorer"]
)

st.sidebar.markdown("---")
st.sidebar.info("""
**Superstore Strategy Analysis**

A portfolio-ready data analysis project covering:
- Data Cleaning & QA
- Sales Insights & Trends
- Forecasting (12-month horizon)
- RFM Customer Segmentation
""")

# =============================================================================
# Load All Data
# =============================================================================
@st.cache_data
def get_all_data():
    """Cache and load all data files."""
    return {
        'master': load_data('data_clean/Superstore_Cleaned.xlsx'),
        'monthly_sales': load_first_available(
            'outputs/day2_tables/monthly_sales.csv',
            'outputs/day3_tables/monthly_sales.csv'
        ),
        'region': load_data('outputs/day2_tables/sales_by_region.csv'),
        'segment': load_data('outputs/day2_tables/sales_by_segment.csv'),
        'category': load_data('outputs/day2_tables/sales_by_category.csv'),
        'subcategory': load_data('outputs/day2_tables/sales_by_subcategory.csv'),
        'rfm': load_data('outputs/day3_rfm_segments.csv'),
        'rfm_summary': load_data('outputs/day3_tables/rfm_segment_summary.csv'),
        'forecast': load_data('outputs/day3_forecast.csv'),
        'yearly': load_data('outputs/day2_tables/yearly_sales.csv'),
        'shipping': load_data('outputs/day2_tables/shipping_mode_summary.csv'),
    }

data = get_all_data()

# =============================================================================
# PAGE 1: Executive Summary
# =============================================================================
if page == "🏠 Executive Summary":
    st.markdown('<div class="main-header">📊 Superstore Strategy Analysis</div>', unsafe_allow_html=True)
    st.markdown("*End-to-end sales, operations & retention analysis*")
    
    # Key Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_sales = 2261536.78  # From QA report
        if data['monthly_sales'] is not None:
            total_sales = data['monthly_sales']['Sales'].sum()
        st.metric("Total Sales", format_currency(total_sales))
    
    with col2:
        st.metric("Total Orders", "9,800")
    
    with col3:
        st.metric("Date Range", "2015-2018")
    
    with col4:
        if data['rfm'] is not None:
            n_customers = data['rfm']['Customer ID'].nunique()
            st.metric("Unique Customers", f"{n_customers:,}")
        else:
            st.metric("Unique Customers", "~800")
    
    st.markdown("---")
    
    # Key Findings
    st.markdown('<div class="sub-header">🎯 Key Findings</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="insight-box">', unsafe_allow_html=True)
        st.markdown("**📍 Top Region:** West (31.4% of sales)")
        st.markdown("**👥 Top Segment:** Consumer (50.8% of sales)")
        st.markdown("**📦 Top Category:** Technology (36.6% of sales)")
        st.markdown("**🏷️ Top Sub-Category:** Phones ($327,782)")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="insight-box">', unsafe_allow_html=True)
        st.markdown("**🚚 Shipping:** Same Day (0 days) vs Standard Class (5 days)")
        st.markdown("**👤 Customer Concentration:** Top 20% = 48.3% of sales")
        st.markdown("**📈 Forecast:** +8.3% growth projected")
        st.markdown("**🎯 Model:** Trend + Month Seasonality (MAPE: 21%)")
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Recommendations
    st.markdown('<div class="sub-header">💡 Recommendations</div>', unsafe_allow_html=True)
    
    rec_col1, rec_col2, rec_col3 = st.columns(3)
    
    with rec_col1:
        st.markdown("**🔄 Retention**")
        st.markdown("Prioritize At-Risk high-value customers with reactivation offers and targeted outreach campaigns.")
    
    with rec_col2:
        st.markdown("**📈 Growth**")
        st.markdown("Upsell/bundle offers for Loyal + Champions segments in top-performing sub-categories.")
    
    with rec_col3:
        st.markdown("**⚙️ Operations**")
        st.markdown("Reduce Ship Days for high-value segments. Set shipping SLA targets by Ship Mode.")
    
    st.markdown("---")
    
    # Full Executive Summary
    with st.expander("📄 View Full Executive Summary"):
        exec_summary = load_markdown('outputs/day5_executive_summary.md')
        st.markdown(exec_summary)

# =============================================================================
# PAGE 2: Sales Insights
# =============================================================================
elif page == "📈 Sales Insights":
    st.markdown('<div class="main-header">📈 Sales Insights</div>', unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "📍 By Region", "👥 By Segment", "📦 By Category"])
    
    # Tab 1: Overview
    with tab1:
        if data['monthly_sales'] is not None:
            df_monthly = data['monthly_sales'].copy()
            df_monthly['Year-Month'] = pd.to_datetime(df_monthly['Year-Month'])
            
            fig = px.line(
                df_monthly, 
                x='Year-Month', 
                y='Sales',
                title='Monthly Sales Trend (2015-2018)',
                template='plotly_white'
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            # YoY Growth
            if data['yearly'] is not None:
                st.markdown("### 📅 Year-over-Year Growth")
                df_yearly = data['yearly'].copy()
                if 'YoY Growth' in df_yearly.columns:
                    df_yearly['YoY Growth %'] = df_yearly['YoY Growth'] * 100
                    fig_yoy = px.bar(
                        df_yearly,
                        x='Year',
                        y='YoY Growth %',
                        title='Year-over-Year Growth Rate',
                        template='plotly_white',
                        color='YoY Growth %',
                        color_continuous_scale='RdYlGn'
                    )
                    st.plotly_chart(fig_yoy, use_container_width=True)
        else:
            st.warning("Monthly sales data not found. Please run the pipeline first.")
    
    # Tab 2: By Region
    with tab2:
        if data['region'] is not None:
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.pie(
                    data['region'],
                    values='Sales',
                    names='Region',
                    title='Sales by Region',
                    template='plotly_white'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.bar(
                    data['region'],
                    x='Region',
                    y='Sales',
                    title='Sales by Region (Bar)',
                    template='plotly_white',
                    color='Sales',
                    color_continuous_scale='Blues'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("### 📋 Region Details")
            st.dataframe(data['region'], use_container_width=True)
        else:
            st.warning("Region data not found.")
    
    # Tab 3: By Segment
    with tab3:
        if data['segment'] is not None:
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.pie(
                    data['segment'],
                    values='Sales',
                    names='Segment',
                    title='Sales by Segment',
                    template='plotly_white',
                    color_discrete_sequence=px.colors.qualitative.Set2
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.bar(
                    data['segment'],
                    x='Segment',
                    y='Sales',
                    title='Sales by Segment (Bar)',
                    template='plotly_white',
                    color='Sales',
                    color_continuous_scale='Greens'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("### 📋 Segment Details")
            st.dataframe(data['segment'], use_container_width=True)
        else:
            st.warning("Segment data not found.")
    
    # Tab 4: By Category
    with tab4:
        if data['category'] is not None and data['subcategory'] is not None:
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.pie(
                    data['category'],
                    values='Sales',
                    names='Category',
                    title='Sales by Category',
                    template='plotly_white'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                top_sub = data['subcategory'].head(10)
                fig = px.bar(
                    top_sub,
                    x='Sub-Category',
                    y='Sales',
                    title='Top 10 Sub-Categories',
                    template='plotly_white',
                    color='Sales',
                    color_continuous_scale='Purples'
                )
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("### 📋 Category Details")
            st.dataframe(data['category'], use_container_width=True)
        else:
            st.warning("Category data not found.")

# =============================================================================
# PAGE 3: Forecast
# =============================================================================
elif page == "🔮 Forecast":
    st.markdown('<div class="main-header">🔮 Sales Forecast</div>', unsafe_allow_html=True)
    st.markdown("*12-month baseline forecast using Trend + Month Seasonality model*")
    
    if data['monthly_sales'] is not None and data['forecast'] is not None:
        # Prepare data
        df_monthly = data['monthly_sales'].copy()
        df_monthly['Year-Month'] = pd.to_datetime(df_monthly['Year-Month'])
        df_monthly['Type'] = 'Actual'
        
        df_forecast = data['forecast'].copy()
        df_forecast['Year-Month'] = pd.to_datetime(df_forecast['Year-Month'])
        df_forecast['Type'] = 'Forecast'
        df_forecast = df_forecast.rename(columns={'Forecast_Sales': 'Sales'})
        
        # Connect forecast to actual data (add last actual point to forecast line)
        last_actual = df_monthly.iloc[-1:][['Year-Month', 'Sales']].copy()
        last_actual['Type'] = 'Forecast'
        df_forecast_connected = pd.concat([last_actual, df_forecast[['Year-Month', 'Sales', 'Type']]], ignore_index=True)
        
        # Forecast chart
        fig = go.Figure()
        
        # Actual data
        fig.add_trace(go.Scatter(
            x=df_monthly['Year-Month'],
            y=df_monthly['Sales'],
            mode='lines',
            name='Actual',
            line=dict(color='#1f77b4', width=2)
        ))
        
        # Forecast data (connected to actual)
        fig.add_trace(go.Scatter(
            x=df_forecast_connected['Year-Month'],
            y=df_forecast_connected['Sales'],
            mode='lines',
            name='Forecast',
            line=dict(color='#ff7f0e', width=2, dash='dash')
        ))
        
        fig.update_layout(
            title='Monthly Sales: Actual + 12-Month Forecast',
            template='plotly_white',
            height=500,
            xaxis_title='Month',
            yaxis_title='Sales ($)',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Forecast metrics
        col1, col2, col3 = st.columns(3)
        
        last_12_actual = df_monthly.tail(12)['Sales'].sum()
        next_12_forecast = df_forecast['Sales'].sum()
        growth = (next_12_forecast - last_12_actual) / last_12_actual * 100
        
        with col1:
            st.metric("Last 12 Months (Actual)", format_currency(last_12_actual))
        with col2:
            st.metric("Next 12 Months (Forecast)", format_currency(next_12_forecast))
        with col3:
            st.metric("Projected Growth", f"{growth:+.1f}%", delta=f"{growth:+.1f}%")
        
        # Forecast table
        st.markdown("### 📋 Forecast Details")
        st.dataframe(df_forecast[['Year-Month', 'Sales', 'Model']], use_container_width=True)
        
        # Model performance
        st.markdown("### 📊 Model Performance")
        st.info("**Selected Model:** Trend + Month Seasonality  ")
        st.info("**Rolling MAPE (3-month horizon):** 21.0%  ")
        st.info("This model outperforms Linear Trend and Seasonal Naive baselines on backtesting.")
        
    else:
        st.warning("Forecast data not found. Please run the pipeline first.")

# =============================================================================
# PAGE 4: RFM Segmentation
# =============================================================================
elif page == "👥 RFM Segmentation":
    st.markdown('<div class="main-header">👥 RFM Customer Segmentation</div>', unsafe_allow_html=True)
    st.markdown("*Recency, Frequency, Monetary analysis for customer retention strategy*")
    
    if data['rfm'] is not None and data['rfm_summary'] is not None:
        # RFM Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Customers", f"{len(data['rfm']):,}")
        with col2:
            avg_monetary = data['rfm']['Monetary'].mean()
            st.metric("Avg Customer Value", format_currency(avg_monetary))
        with col3:
            avg_recency = data['rfm']['RecencyDays'].mean()
            st.metric("Avg Recency", f"{avg_recency:.0f} days")
        with col4:
            avg_frequency = data['rfm']['Frequency'].mean()
            st.metric("Avg Frequency", f"{avg_frequency:.1f} orders")
        
        st.markdown("---")
        
        # Segment distribution
        st.markdown("### 📊 Segment Distribution")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.pie(
                data['rfm_summary'],
                values='Customers',
                names='Segment',
                title='Customers by Segment',
                template='plotly_white'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.bar(
                data['rfm_summary'],
                x='Segment',
                y='TotalMonetary',
                title='Total Revenue by Segment',
                template='plotly_white',
                color='TotalMonetary',
                color_continuous_scale='RdYlGn'
            )
            fig.update_layout(xaxis_tickangle=-30)
            st.plotly_chart(fig, use_container_width=True)
        
        # Segment details table
        st.markdown("### 📋 Segment Details")
        st.dataframe(data['rfm_summary'], use_container_width=True)
        
        # RFM Explorer
        st.markdown("---")
        st.markdown("### 🔍 RFM Customer Explorer")
        
        # Filters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            selected_segment = st.multiselect(
                "Filter by Segment:",
                options=data['rfm']['Segment'].unique(),
                default=[]
            )
        
        with col2:
            min_monetary = st.number_input(
                "Min Monetary Value:",
                value=0.0,
                step=100.0
            )
        
        with col3:
            max_recency = st.number_input(
                "Max Recency (days):",
                value=365,
                step=30
            )
        
        # Apply filters
        df_filtered = data['rfm'].copy()
        if selected_segment:
            df_filtered = df_filtered[df_filtered['Segment'].isin(selected_segment)]
        df_filtered = df_filtered[df_filtered['Monetary'] >= min_monetary]
        df_filtered = df_filtered[df_filtered['RecencyDays'] <= max_recency]
        
        st.markdown(f"**Showing {len(df_filtered)} customers**")
        st.dataframe(df_filtered, use_container_width=True)
        
        # RFM 3D Scatter
        st.markdown("### 🎯 RFM 3D Visualization")
        
        sample_size = min(500, len(data['rfm']))
        df_sample = data['rfm'].sample(sample_size) if len(data['rfm']) > sample_size else data['rfm']
        
        fig = px.scatter_3d(
            df_sample,
            x='RecencyDays',
            y='Frequency',
            z='Monetary',
            color='Segment',
            size='RFM_Score',
            title='RFM 3D Scatter Plot (Sample)',
            template='plotly_white'
        )
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
        
    else:
        st.warning("RFM data not found. Please run the pipeline first.")

# =============================================================================
# PAGE 5: Data Explorer
# =============================================================================
elif page == "📦 Data Explorer":
    st.markdown('<div class="main-header">📦 Data Explorer</div>', unsafe_allow_html=True)
    
    if data['master'] is not None:
        st.markdown(f"**Dataset:** {len(data['master']):,} rows × {len(data['master'].columns)} columns")
        
        # Column selector
        selected_cols = st.multiselect(
            "Select columns to display:",
            options=list(data['master'].columns),
            default=['Order Date', 'Customer ID', 'Region', 'Segment', 'Category', 'Sales']
        )
        
        # Filters
        col1, col2 = st.columns(2)
        
        with col1:
            if 'Region' in data['master'].columns:
                region_filter = st.multiselect(
                    "Filter by Region:",
                    options=data['master']['Region'].unique(),
                    default=[]
                )
        
        with col2:
            if 'Segment' in data['master'].columns:
                segment_filter = st.multiselect(
                    "Filter by Segment:",
                    options=data['master']['Segment'].unique(),
                    default=[]
                )
        
        # Apply filters
        df_display = data['master'][selected_cols].copy() if selected_cols else data['master'].copy()
        
        if region_filter and 'Region' in df_display.columns:
            df_display = df_display[df_display['Region'].isin(region_filter)]
        if segment_filter and 'Segment' in df_display.columns:
            df_display = df_display[df_display['Segment'].isin(segment_filter)]
        
        st.markdown(f"**Displaying {len(df_display):,} rows**")
        st.dataframe(df_display, use_container_width=True)
        
        # Download button
        csv = df_display.to_csv(index=False)
        st.download_button(
            label="📥 Download as CSV",
            data=csv,
            file_name="superstore_filtered.csv",
            mime="text/csv"
        )
        
    else:
        st.warning("Master data not found. Please run the pipeline first.")

# =============================================================================
# Footer
# =============================================================================
st.markdown("---")
st.markdown("<center>📊 Superstore Strategy Analysis Dashboard | Built with Streamlit</center>", unsafe_allow_html=True)
