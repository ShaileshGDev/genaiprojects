
# Quick Start Guide - Superstore Dashboard

## 📋 Overview

This guide will help you get the Superstore Analytics Dashboard up and running in minutes.

---

## ⚡ 5-Minute Quick Start (Local)

### Step 1: Prerequisites
```bash
# Python 3.8+ required
python --version

# Verify pip
pip --version
```

### Step 2: Install Dependencies
```bash
# Create virtual environment (optional but recommended)
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install required packages
pip install streamlit pandas numpy plotly
```

### Step 3: Prepare Files
```bash
# Create project directory
mkdir superstore-dashboard
cd superstore-dashboard

# Copy these files to the directory:
# - streamlit-dashboard.py
# - Sample-Superstore.csv
```

### Step 4: Run the Dashboard
```bash
streamlit run streamlit-dashboard.py
```

**Output:**
```
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.x.x:8501
```

### Step 5: Access the Dashboard
- Open browser: `http://localhost:8501`
- Interact with filters in the sidebar
- View real-time charts and metrics
- Download filtered data as CSV

---

## 🐳 Docker Quick Start (2 Minutes)

### Prerequisites
- Docker installed: https://docs.docker.com/get-docker/

### Run in Docker
```bash
# Create Dockerfile in project directory
cat > Dockerfile << 'EOF'
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "streamlit-dashboard.py", "--server.port=8501"]
EOF

# Create requirements.txt
cat > requirements.txt << 'EOF'
streamlit==1.28.1
pandas==2.1.1
numpy==1.24.3
plotly==5.17.0
EOF

# Build and run
docker build -t superstore-dashboard .
docker run -p 8501:8501 superstore-dashboard

# Access: http://localhost:8501
```

---

## ☁️ Deploy to Streamlit Cloud (3 Minutes)

### Step 1: Push to GitHub
```bash
# Create GitHub repository
# Push your files:
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/your-username/superstore-dashboard.git
git push -u origin main
```

### Step 2: Deploy to Streamlit Cloud
1. Go to: https://share.streamlit.io
2. Sign in with GitHub
3. Click "New app"
4. Select your repository and main file
5. Click "Deploy"

**Your dashboard is live!** Share the public URL.

---

## 🔧 Common Tasks

### Change CSV File Location
Edit `streamlit-dashboard.py`:
```python
# Line to modify:
df = pd.read_csv('path/to/your/file.csv', encoding='latin-1')
```

### Customize Colors
Edit theme in `streamlit-dashboard.py`:
```python
st.set_page_config(
    page_title="Your Title",
    page_icon="📊",
    # ... other settings
)
```

### Add New Metrics
Add after line ~150:
```python
with col5:
    total_customers = filtered_df['Customer ID'].nunique()
    st.metric(label="👥 Total Customers", value=f"{total_customers:,}")
```

### Change Date Range Preset
Edit line ~90:
```python
default_end = pd.Timestamp.now()
default_start = default_end - pd.Timedelta(days=90)  # Change 90 to your preference

date_range = st.sidebar.date_input(
    "Select Date Range",
    value=(default_start, default_end),
    # ...
)
```

---

## 📊 Dashboard Features

### Filters (Sidebar)
- **Date Range:** Select custom date period
- **Categories:** Choose product categories
- **Segments:** Filter by customer segment
- **Regions:** Select geographic regions

### Key Metrics (Top Section)
- Total Sales
- Total Profit & Margin %
- Total Orders
- Average Order Value

### Visualizations
1. **Sales Trend** - Line chart of sales over time
2. **Profit Trend** - Profit trajectory
3. **Category Analysis** - Sales & profit by category
4. **Segment Pie Chart** - Sales distribution by customer type
5. **Regional Analysis** - Performance by region
6. **Top Products** - Best sellers
7. **Sub-Category Breakdown** - Detailed category analysis
8. **Discount Impact** - Relationship between discount and profit
9. **Category vs Segment Heatmap** - Cross-analysis
10. **Profit Distribution** - Histogram of profit values

### Data Table
- View detailed transactions
- Select columns to display
- Sort and filter
- Download as CSV

### Summary Statistics
- Customer and location metrics
- Discount and quantity averages
- Order value ranges
- Profitable order count

---

## 🐛 Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'streamlit'"
**Solution:**
```bash
pip install streamlit
# Or reinstall all requirements
pip install -r requirements.txt
```

### Issue: "File not found: Sample-Superstore.csv"
**Solution:**
- Ensure CSV file is in the same directory as the script
- Or use full path: `/path/to/Sample-Superstore.csv`

### Issue: "Port 8501 already in use"
**Solution:**
```bash
# Use a different port
streamlit run streamlit-dashboard.py --server.port 8502
```

### Issue: Dashboard runs slow with large CSV
**Solution:**
```python
# Add data sampling in streamlit-dashboard.py after line ~30:
# Load only recent data
df = df[df['Order Date'] >= '2015-01-01']  # Reduce date range
```

### Issue: Plotly charts not displaying
**Solution:**
```bash
pip install --upgrade plotly
streamlit cache clear
streamlit run streamlit-dashboard.py
```

---

## 📈 Performance Tips

### For Large Datasets (>100K rows)
```python
# 1. Increase caching TTL (more memory, less recomputation)
@st.cache_data(ttl=7200)  # 2 hours
def load_data():
    return pd.read_csv('data.csv')

# 2. Use data sampling for display
df_display = df.sample(min(10000, len(df)))

# 3. Add progress indicator
with st.spinner("Loading data..."):
    df = load_data()
```

### For Frequent Dashboard Visitors
```bash
# Run with higher resource allocation
streamlit run app.py \
  --logger.level=warning \
  --client.showErrorDetails=false \
  --server.enableCORS=false
```

---

## 🔒 Security Best Practices

### 1. Protect Sensitive Data
```python
# Never commit credentials to GitHub
# Use environment variables instead:
import os
api_key = os.getenv('API_KEY')

# Or use Streamlit secrets
db_password = st.secrets["database"]["password"]
```

### 2. Set Up .gitignore
```bash
# Create .gitignore file
echo "*.csv" > .gitignore
echo ".env" >> .gitignore
echo "venv/" >> .gitignore
echo "__pycache__/" >> .gitignore
```

### 3. Enable SSL for Production
- Use HTTPS/TLS for all connections
- Obtain certificate: Let's Encrypt (free)
- Configure in Nginx or load balancer

---

## 📚 Next Steps

1. **Add Authentication:** See `advanced-config.md` for login setup
2. **Connect to Database:** Replace CSV with PostgreSQL
3. **Deploy to Cloud:** Use AWS, GCP, or Azure
4. **Customize Branding:** Add company logo and colors
5. **Add Alerts:** Get notified of KPI changes
6. **Monitor Performance:** Set up logging and metrics

---

## 📖 Documentation Files

- **streamlit-dashboard.py** - Main application code
- **deployment-guide.md** - Complete deployment guide
- **advanced-config.md** - Enterprise configurations
- **requirements.txt** - Python dependencies

---

## 🆘 Need Help?

### Resources
- **Streamlit Docs:** https://docs.streamlit.io
- **Plotly Docs:** https://plotly.com/python
- **Stack Overflow:** Tag with `streamlit`
- **GitHub Issues:** Report bugs in repo

### Common Questions

**Q: Can I use with my own data?**
A: Yes! Replace `Sample-Superstore.csv` with your CSV file.

**Q: Can I add more charts?**
A: Yes! Follow the Plotly examples in the code. Add any chart in seconds.

**Q: Can I deploy to my company's servers?**
A: Yes! Follow Docker or AWS deployment guides in deployment-guide.md

**Q: Can multiple users use it simultaneously?**
A: Yes! Each user gets their own session. Streamlit handles concurrency.

**Q: Can I modify the dashboard?**
A: Absolutely! It's fully customizable Python code. Modify as needed.

---

## 🚀 Quick Command Reference

```bash
# Start dashboard locally
streamlit run streamlit-dashboard.py

# Clear cache (if something looks wrong)
streamlit cache clear

# Run on specific port
streamlit run streamlit-dashboard.py --server.port 8502

# Deploy to Streamlit Cloud
git push origin main  # Auto-deploys if connected

# Run in Docker
docker build -t dashboard . && docker run -p 8501:8501 dashboard

# SSH into server and pull latest
git pull origin main
systemctl restart streamlit
```

---

## 📞 Support

For issues or questions:
1. Check this Quick Start Guide
2. Review advanced-config.md
3. Check Streamlit documentation
4. Review deployment-guide.md for your platform

---

**Happy analyzing! 📊**

Last updated: December 2025
