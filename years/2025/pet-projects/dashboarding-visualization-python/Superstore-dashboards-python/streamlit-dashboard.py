"""
Enterprise Superstore Dashboard - PROFESSIONAL PALETTE
Built with Streamlit for interactive data visualization and analysis
Palette: Deep Navy, Teal, Cool Gray, Dark Gray, Soft Amber
Fonts: Montserrat (headings), Source Sans Pro (body)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings

warnings.filterwarnings('ignore')

# Configure Streamlit
st.set_page_config(
    page_title="Superstore Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with Professional Palette
# Colors: Deep Navy (#12355B), Teal (#1B9AAA), Cool Gray (#E5E7EB), Dark Gray (#111827), Soft Amber (#F5A623)
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;500;600;700&family=Source+Sans+Pro:wght@400;500;600;700&display=swap');
    
    /* Root Variables - Professional Palette */
    :root {
        --deep-navy: #12355B;
        --teal: #1B9AAA;
        --cool-gray: #E5E7EB;
        --dark-gray: #111827;
        --soft-amber: #F5A623;
        --white: #FFFFFF;
        --light-bg: #F9FAFB;
    }
    
    /* Body and General Styling */
    * {
        font-family: 'Source Sans Pro', sans-serif;
    }
    
    body {
        background-color: #F9FAFB;
    }
    
    /* Main Container */
    .main {
        padding-top: 2rem;
        background-color: #F9FAFB;
    }
    
    /* Headers - Montserrat Font */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Montserrat', sans-serif;
        font-weight: 600;
        color: #111827;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(135deg, #12355B 0%, #0d1e35 100%);
        color: white;
    }
    
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        color: white;
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        color: white;
    }
    
    /* Main Header Title */
    .header-title {
        color: #12355B;
        font-family: 'Montserrat', sans-serif;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 1rem;
        letter-spacing: -0.5px;
    }
    
    .header-subtitle {
        color: #1B9AAA;
        font-family: 'Source Sans Pro', sans-serif;
        font-size: 1rem;
        font-weight: 500;
        margin-bottom: 2rem;
    }
    
    /* Metric Cards */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, #FFFFFF 0%, #F9FAFB 100%);
        border-left: 5px solid #1B9AAA;
        border-radius: 8px;
        padding: 20px;
        box-shadow: 0 2px 8px rgba(18, 53, 91, 0.08);
    }
    
    /* Section Headers */
    .section-header {
        color: #1B9AAA;
        font-family: 'Montserrat', sans-serif;
        font-size: 1.3rem;
        font-weight: 600;
        border-left: 4px solid #F5A623;
        padding-left: 12px;
        margin: 20px 0 15px 0;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #12355B 0%, #0d1e35 100%);
        color: white;
        border-radius: 6px;
        padding: 10px 20px;
        font-family: 'Source Sans Pro', sans-serif;
        font-weight: 600;
        border: none;
        transition: all 0.3s ease;
        box-shadow: 0 2px 6px rgba(18, 53, 91, 0.15);
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #0d1e35 0%, #061024 100%);
        box-shadow: 0 4px 12px rgba(18, 53, 91, 0.25);
        transform: translateY(-2px);
    }
    
    /* Download Button */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #1B9AAA 0%, #157a87 100%);
        color: white;
        font-weight: 600;
    }
    
    .stDownloadButton > button:hover {
        background: linear-gradient(135deg, #157a87 0%, #0f5863 100%);
    }
    
    /* Select/Multiselect - Professional Colors */
    [data-testid="stSelectbox"] {
        color: #12355B;
    }
    
    [data-testid="stMultiSelect"] {
        color: #12355B;
    }
    
    /* Input Fields */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stDateInput > div > div > input {
        border: 2px solid #E5E7EB;
        border-radius: 6px;
        padding: 10px;
        font-family: 'Source Sans Pro', sans-serif;
        color: #111827;
    }
    
    .stTextInput > div > div > input:focus,
    .stNumberInput > div > div > input:focus,
    .stDateInput > div > div > input:focus {
        border-color: #12355B;
        box-shadow: 0 0 0 3px rgba(18, 53, 91, 0.1);
    }
    
    /* Divider */
    hr {
        border-color: #E5E7EB;
        margin: 2rem 0;
    }
    
    /* Data Table */
    .stDataFrame {
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 2px 8px rgba(18, 53, 91, 0.08);
    }
    
    /* Chart Container */
    [data-testid="stPlotlyContainer"] {
        border-radius: 8px;
        padding: 15px;
        background-color: white;
        box-shadow: 0 2px 8px rgba(18, 53, 91, 0.08);
        margin: 15px 0;
    }
    
    /* Tab Styling */
    [data-testid="stTabs"] [aria-selected="true"] {
        border-bottom: 3px solid #12355B;
        color: #12355B;
    }
    
    [data-testid="stTabs"] [aria-selected="true"] button {
        color: #12355B;
        font-weight: 600;
    }
    
    /* Expander */
    [data-testid="stExpander"] {
        border-color: #E5E7EB;
    }
    
    [data-testid="stExpander"] button {
        color: #12355B;
        font-weight: 600;
        font-family: 'Montserrat', sans-serif;
    }
    
    /* Error/Success Messages */
    .stAlert {
        border-radius: 6px;
        font-family: 'Source Sans Pro', sans-serif;
    }
    
    .stSuccess {
        background-color: rgba(27, 154, 170, 0.1);
        border-color: #1B9AAA;
        color: #157a87;
    }
    
    .stError {
        background-color: rgba(245, 166, 35, 0.1);
        border-color: #F5A623;
        color: #c97a15;
    }
    
    .stWarning {
        background-color: rgba(245, 166, 35, 0.15);
        border-color: #F5A623;
        color: #c97a15;
    }
    
    /* Info Box Styling */
    .info-box {
        background: linear-gradient(135deg, #12355B 0%, #0d1e35 100%);
        color: white;
        padding: 20px;
        border-radius: 8px;
        margin: 15px 0;
        font-family: 'Source Sans Pro', sans-serif;
    }
    
    .info-box h3 {
        color: #F5A623;
        margin-top: 0;
    }
    
    /* Footer Styling */
    .footer {
        color: #9CA3AF;
        text-align: center;
        padding: 20px;
        margin-top: 40px;
        border-top: 2px solid #E5E7EB;
        font-size: 0.9rem;
    }
    
    /* Link Colors */
    a {
        color: #12355B;
        text-decoration: none;
    }
    
    a:hover {
        color: #1B9AAA;
    }
    </style>
""", unsafe_allow_html=True)

# Load Data with Caching
@st.cache_data
def load_data():
    df = pd.read_csv('Sample-Superstore.csv', encoding='latin-1')
    df['Order Date'] = pd.to_datetime(df['Order Date'])
    df['Ship Date'] = pd.to_datetime(df['Ship Date'])
    df['Year'] = df['Order Date'].dt.year
    df['Month'] = df['Order Date'].dt.month
    df['Quarter'] = df['Order Date'].dt.quarter
    df['Month_Name'] = df['Order Date'].dt.strftime('%B')
    return df

df = load_data()

# Sidebar Configuration
st.sidebar.markdown("### 📊 Dashboard Filters")

# Date Range Filter
date_range = st.sidebar.date_input(
    "Select Date Range",
    value=(df['Order Date'].min(), df['Order Date'].max()),
    min_value=df['Order Date'].min(),
    max_value=df['Order Date'].max()
)

# Category Filter
categories = st.sidebar.multiselect(
    "Select Categories",
    options=df['Category'].unique(),
    default=df['Category'].unique()
)

# Segment Filter
segments = st.sidebar.multiselect(
    "Select Customer Segments",
    options=df['Segment'].unique(),
    default=df['Segment'].unique()
)

# Region Filter
regions = st.sidebar.multiselect(
    "Select Regions",
    options=df['Region'].unique(),
    default=df['Region'].unique()
)

# Apply Filters
filtered_df = df[
    (df['Order Date'] >= pd.Timestamp(date_range[0])) &
    (df['Order Date'] <= pd.Timestamp(date_range[1])) &
    (df['Category'].isin(categories)) &
    (df['Segment'].isin(segments)) &
    (df['Region'].isin(regions))
]

# Header
st.markdown("<div class='header-title'>📊 Superstore Analytics Dashboard</div>", unsafe_allow_html=True)
st.markdown("<div class='header-subtitle'>Enterprise-grade insights into sales performance</div>", unsafe_allow_html=True)
st.markdown(f"*Last Updated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}*")

# Key Metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    total_sales = filtered_df['Sales'].sum()
    st.metric(label="💰 Total Sales", value=f"${total_sales:,.0f}")

with col2:
    total_profit = filtered_df['Profit'].sum()
    profit_margin = (total_profit / total_sales * 100) if total_sales > 0 else 0
    st.metric(label="📈 Total Profit", value=f"${total_profit:,.0f}", delta=f"{profit_margin:.1f}% margin")

with col3:
    total_orders = filtered_df['Order ID'].nunique()
    st.metric(label="📦 Total Orders", value=f"{total_orders:,}")

with col4:
    avg_order_value = filtered_df['Sales'].sum() / total_orders if total_orders > 0 else 0
    st.metric(label="🎯 Avg Order Value", value=f"${avg_order_value:,.0f}")

st.divider()

# Row 1: Sales and Profit Trends
st.markdown("<div class='section-header'>📈 Sales & Profit Performance</div>", unsafe_allow_html=True)
col1, col2 = st.columns(2)

with col1:
    # Sales Trend
    sales_by_date = filtered_df.groupby(filtered_df['Order Date'].dt.date)['Sales'].sum().reset_index()
    fig_sales = px.line(
        sales_by_date,
        x='Order Date',
        y='Sales',
        title='Sales Trend Over Time',
        markers=True,
        template='plotly_white'
    )
    fig_sales.update_traces(line=dict(color='#12355B', width=3), marker=dict(size=6, color='#F5A623'))
    fig_sales.update_layout(hovermode='x unified', height=400, font=dict(family="Source Sans Pro", color="#111827"))
    fig_sales.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#E5E7EB')
    fig_sales.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#E5E7EB')
    st.plotly_chart(fig_sales, use_container_width=True)

with col2:
    # Profit Trend
    profit_by_date = filtered_df.groupby(filtered_df['Order Date'].dt.date)['Profit'].sum().reset_index()
    fig_profit = px.line(
        profit_by_date,
        x='Order Date',
        y='Profit',
        title='Profit Trend Over Time',
        markers=True,
        template='plotly_white'
    )
    fig_profit.update_traces(line=dict(color='#1B9AAA', width=3), marker=dict(size=6, color='#F5A623'))
    fig_profit.update_layout(hovermode='x unified', height=400, font=dict(family="Source Sans Pro", color="#111827"))
    fig_profit.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#E5E7EB')
    fig_profit.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#E5E7EB')
    st.plotly_chart(fig_profit, use_container_width=True)

# Row 2: Category and Segment Analysis
st.markdown("<div class='section-header'>🏪 Category & Customer Segment Analysis</div>", unsafe_allow_html=True)
col1, col2 = st.columns(2)

with col1:
    # Sales by Category
    category_sales = filtered_df.groupby('Category')[['Sales', 'Profit']].sum().reset_index()
    fig_category = px.bar(
        category_sales,
        x='Category',
        y=['Sales', 'Profit'],
        title='Sales and Profit by Category',
        barmode='group',
        template='plotly_white',
        color_discrete_map={'Sales': '#12355B', 'Profit': '#1B9AAA'}
    )
    fig_category.update_layout(height=400, hovermode='x unified', font=dict(family="Source Sans Pro", color="#111827"))
    st.plotly_chart(fig_category, use_container_width=True)

with col2:
    # Sales by Segment
    segment_sales = filtered_df.groupby('Segment')[['Sales', 'Profit']].sum().reset_index()
    fig_segment = px.pie(
        segment_sales,
        names='Segment',
        values='Sales',
        title='Sales Distribution by Customer Segment',
        template='plotly_white',
        color_discrete_sequence=['#12355B', '#1B9AAA', '#E5E7EB']
    )
    fig_segment.update_layout(height=400, font=dict(family="Source Sans Pro", color="#111827"))
    st.plotly_chart(fig_segment, use_container_width=True)

# Row 3: Regional and Product Analysis
st.markdown("<div class='section-header'>🌍 Regional Performance & Top Products</div>", unsafe_allow_html=True)
col1, col2 = st.columns(2)

with col1:
    # Sales by Region
    region_sales = filtered_df.groupby('Region')[['Sales', 'Profit', 'Order ID']].agg({
        'Sales': 'sum',
        'Profit': 'sum',
        'Order ID': 'nunique'
    }).reset_index().rename(columns={'Order ID': 'Orders'})
    
    fig_region = px.bar(
        region_sales,
        x='Region',
        y='Sales',
        color='Profit',
        title='Sales by Region',
        template='plotly_white',
        color_continuous_scale=['#12355B', '#F5A623']
    )
    fig_region.update_layout(height=400, hovermode='x unified', font=dict(family="Source Sans Pro", color="#111827"))
    st.plotly_chart(fig_region, use_container_width=True)

with col2:
    # Top 10 Products by Sales
    top_products = filtered_df.groupby('Product Name')[['Sales', 'Profit']].sum().reset_index()
    top_products = top_products.nlargest(10, 'Sales')
    
    fig_products = px.bar(
        top_products,
        x='Sales',
        y='Product Name',
        color='Profit',
        title='Top 10 Products by Sales',
        template='plotly_white',
        color_continuous_scale=['#12355B', '#F5A623'],
        orientation='h'
    )
    fig_products.update_layout(height=400, margin=dict(l=250), font=dict(family="Source Sans Pro", color="#111827"))
    st.plotly_chart(fig_products, use_container_width=True)

# Row 4: Sub-Category and Discount Analysis
st.markdown("<div class='section-header'>📊 Sub-Category Performance & Discount Impact</div>", unsafe_allow_html=True)
col1, col2 = st.columns(2)

with col1:
    # Sales by Sub-Category
    subcategory_sales = filtered_df.groupby('Sub-Category')[['Sales', 'Profit']].sum().reset_index()
    subcategory_sales = subcategory_sales.sort_values('Sales', ascending=True).tail(15)
    
    fig_subcat = px.bar(
        subcategory_sales,
        x='Sales',
        y='Sub-Category',
        color='Profit',
        title='Top 15 Sub-Categories by Sales',
        template='plotly_white',
        color_continuous_scale=['#12355B', '#F5A623'],
        orientation='h'
    )
    fig_subcat.update_layout(height=400, font=dict(family="Source Sans Pro", color="#111827"))
    st.plotly_chart(fig_subcat, use_container_width=True)

with col2:
    # Discount vs Profit Analysis
    fig_discount = px.scatter(
        filtered_df,
        x='Discount',
        y='Profit',
        color='Sales',
        size='Quantity',
        title='Discount vs Profit Analysis',
        template='plotly_white',
        color_continuous_scale=['#12355B', '#F5A623']
    )
    fig_discount.update_layout(height=400, font=dict(family="Source Sans Pro", color="#111827"))
    st.plotly_chart(fig_discount, use_container_width=True)

# Row 5: Heatmap and Distribution
st.markdown("<div class='section-header'>🔥 Advanced Analytics</div>", unsafe_allow_html=True)
col1, col2 = st.columns(2)

with col1:
    # Sales Heatmap by Category and Segment
    heatmap_data = filtered_df.pivot_table(
        values='Sales',
        index='Category',
        columns='Segment',
        aggfunc='sum'
    )
    
    fig_heatmap = px.imshow(
        heatmap_data,
        title='Sales Heatmap: Category vs Segment',
        labels=dict(x="Segment", y="Category", color="Sales"),
        template='plotly_white',
        color_continuous_scale=['#F9FAFB', '#F5A623', '#12355B']
    )
    fig_heatmap.update_layout(height=400, font=dict(family="Source Sans Pro", color="#111827"))
    st.plotly_chart(fig_heatmap, use_container_width=True)

with col2:
    # Profit Distribution Histogram
    fig_hist = px.histogram(
        filtered_df,
        x='Profit',
        nbins=50,
        title='Profit Distribution',
        template='plotly_white',
        color_discrete_sequence=['#1B9AAA']
    )
    fig_hist.update_layout(height=400, hovermode='x unified', font=dict(family="Source Sans Pro", color="#111827"))
    st.plotly_chart(fig_hist, use_container_width=True)

st.divider()

# Detailed Data Table
st.markdown("<div class='section-header'>📋 Detailed Sales Data</div>", unsafe_allow_html=True)

table_cols = st.multiselect(
    "Select columns to display",
    options=['Order ID', 'Order Date', 'Customer Name', 'Category', 'Sub-Category', 
             'Sales', 'Quantity', 'Discount', 'Profit', 'Region', 'Segment'],
    default=['Order ID', 'Order Date', 'Customer Name', 'Category', 'Sales', 'Profit']
)

display_df = filtered_df[table_cols].sort_values('Order Date', ascending=False).head(100)
st.dataframe(display_df, use_container_width=True, height=400)

# Download Data
csv = filtered_df.to_csv(index=False)
st.download_button(
    label="📥 Download Filtered Data as CSV",
    data=csv,
    file_name=f"superstore_data_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
    mime="text/csv"
)

st.divider()

# Summary Statistics Section
st.markdown("<div class='section-header'>📊 Summary Statistics</div>", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Customers", filtered_df['Customer ID'].nunique())
    st.metric("Total Cities", filtered_df['City'].nunique())
    st.metric("Total States", filtered_df['State'].nunique())

with col2:
    st.metric("Avg Discount Rate", f"{filtered_df['Discount'].mean():.1%}")
    st.metric("Avg Quantity per Order", f"{filtered_df['Quantity'].mean():.1f}")
    st.metric("Max Order Value", f"${filtered_df['Sales'].max():,.0f}")

with col3:
    st.metric("Min Order Value", f"${filtered_df['Sales'].min():.2f}")
    st.metric("Median Order Value", f"${filtered_df['Sales'].median():,.0f}")
    st.metric("Profitable Orders", f"{(filtered_df['Profit'] > 0).sum():,}")

# Footer
st.divider()
st.markdown("""
<div class='footer'>
Enterprise Dashboard | Built with Streamlit | Professional Palette | Data Analytics
</div>
""", unsafe_allow_html=True)
