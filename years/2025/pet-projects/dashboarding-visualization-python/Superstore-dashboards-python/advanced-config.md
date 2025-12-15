
# Enterprise Dashboard - Configuration & Customization Guide

## Table of Contents
1. Advanced Streamlit Configuration
2. Custom Styling & Branding
3. Data Pipeline Integration
4. Performance Tuning
5. Multi-user Features

---

## 1. ADVANCED STREAMLIT CONFIGURATION

### ~/.streamlit/config.toml (Enterprise Setup)

```toml
[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
font = "sans serif"

[server]
port = 8501
headless = true
runOnSave = false
enableXsrfProtection = true
enableCORS = false
maxUploadSize = 500  # MB
maxMessageSize = 200  # MB
enableWebsocketCompression = true
fileWatcherType = "auto"

[client]
showErrorDetails = false
toolbarMode = "viewer"
showSidebarNavigation = false

[logger]
level = "info"
messageFormat = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"

[browser]
gatherUsageStats = false
serverAddress = "your-domain.com"

[dataFrameSerialization]
enableUnicodeSupport = true

[client.toolbarMode]
mode = "viewer"  # or "developer" for dev mode

[ui]
hideTopBar = false
hideFooter = false
hideSidebarNav = false
```

### Streamlit Secrets (credentials.toml)

Create `~/.streamlit/secrets.toml`:

```toml
# Database Credentials
[database]
host = "your-db-host.rds.amazonaws.com"
port = 5432
user = "admin"
password = "secure_password_here"
database = "superstore"

# API Keys
[api]
openai_key = "sk-..."
datadog_api_key = "dd_..."

# AWS Credentials
[aws]
access_key = "AKIA..."
secret_key = "..."
region = "us-east-1"

# Azure Credentials
[azure]
subscription_id = "..."
tenant_id = "..."
client_id = "..."
client_secret = "..."

# Email Configuration
[email]
smtp_server = "smtp.gmail.com"
smtp_port = 587
sender_email = "alerts@company.com"
sender_password = "..."
```

Access in code: `st.secrets["database"]["host"]`

---

## 2. CUSTOM STYLING & BRANDING

### Advanced Custom CSS

```python
import streamlit as st

# Custom CSS for Enterprise Branding
custom_css = """
    <style>
    /* Company Branding */
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Header Styling */
    .css-1d391kg {
        background-color: #2c3e50;
        padding: 20px;
        border-radius: 10px;
    }
    
    /* Sidebar Customization */
    [data-testid="stSidebar"] {
        background-color: #34495e;
        color: white;
    }
    
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        color: white;
    }
    
    /* Metric Cards */
    [data-testid="metric-container"] {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border-left: 5px solid #667eea;
    }
    
    /* Buttons */
    .stButton > button {
        background-color: #667eea;
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background-color: #764ba2;
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.2);
    }
    
    /* Tables */
    .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    /* Charts Container */
    [data-testid="stPlotlyContainer"] {
        border-radius: 10px;
        padding: 15px;
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
    }
    
    /* Select/Multiselect */
    [data-testid="stSelectbox"], 
    [data-testid="stMultiSelect"] {
        border-radius: 8px;
    }
    
    /* Input Fields */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stDateInput > div > div > input {
        border-radius: 8px;
        border: 2px solid #e9ecef;
        padding: 10px;
    }
    
    /* Tabs */
    [data-testid="stTabs"] [aria-selected="true"] {
        border-bottom: 3px solid #667eea;
    }
    </style>
"""

st.markdown(custom_css, unsafe_allow_html=True)
```

### Company Logo & Branding

```python
import streamlit as st
from PIL import Image

# Add company logo
col1, col2 = st.columns([1, 4])
with col1:
    logo = Image.open("logo.png")
    st.image(logo, width=80)
with col2:
    st.title("🏢 Company Analytics Dashboard")
    st.markdown("*Powered by Data Intelligence*")

# Themed header
st.markdown("""
    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 30px; 
                border-radius: 10px;
                color: white;
                margin-bottom: 30px;'>
        <h2>Welcome to Your Enterprise Dashboard</h2>
        <p>Real-time insights for better decision making</p>
    </div>
""", unsafe_allow_html=True)
```

---

## 3. DATA PIPELINE INTEGRATION

### PostgreSQL Connection Pool

```python
import streamlit as st
import psycopg2
from psycopg2.pool import SimpleConnectionPool
from contextlib import contextmanager

class DatabaseManager:
    _instance = None
    _pool = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    @staticmethod
    @st.cache_resource
    def init_pool():
        return SimpleConnectionPool(
            minconn=1,
            maxconn=20,
            host=st.secrets["database"]["host"],
            port=st.secrets["database"]["port"],
            user=st.secrets["database"]["user"],
            password=st.secrets["database"]["password"],
            database=st.secrets["database"]["database"]
        )
    
    @contextmanager
    def get_connection(self):
        pool = self.init_pool()
        conn = pool.getconn()
        try:
            yield conn
        finally:
            pool.putconn(conn)
    
    def execute_query(self, query, params=None):
        """Execute SELECT query"""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, params or ())
                return pd.DataFrame(
                    cur.fetchall(),
                    columns=[desc[0] for desc in cur.description]
                )
    
    def execute_update(self, query, params=None):
        """Execute INSERT/UPDATE/DELETE"""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, params or ())
                conn.commit()
                return cur.rowcount

# Usage
db = DatabaseManager()
df = db.execute_query(
    "SELECT * FROM sales WHERE order_date > %s",
    (pd.Timestamp.now() - pd.Timedelta(days=30),)
)
```

### Real-time Data with WebSocket

```python
import streamlit as st
import websockets
import json
import asyncio

@st.cache_resource
def init_websocket():
    async def stream_data():
        async with websockets.connect('wss://api.example.com/stream') as websocket:
            while True:
                message = await websocket.recv()
                data = json.loads(message)
                yield data
    return stream_data

# Auto-refresh with placeholder
placeholder = st.empty()
auto_refresh = st.checkbox("Auto-refresh", value=True)

if auto_refresh:
    import time
    with st.spinner("Streaming data..."):
        while True:
            # Simulated real-time data
            new_data = {
                'timestamp': pd.Timestamp.now(),
                'sales': np.random.randint(100, 1000),
                'profit': np.random.randint(10, 200)
            }
            placeholder.metric("Live Sales", f"${new_data['sales']}")
            time.sleep(5)  # Refresh every 5 seconds
```

### API Integration

```python
import requests
import streamlit as st

class APIClient:
    def __init__(self, base_url, api_key):
        self.base_url = base_url
        self.headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
    
    @st.cache_data(ttl=3600)
    def get_sales_data(_self, start_date, end_date):
        """Fetch sales data from API"""
        params = {
            'start_date': start_date,
            'end_date': end_date
        }
        response = requests.get(
            f"{_self.base_url}/sales",
            headers=_self.headers,
            params=params
        )
        response.raise_for_status()
        return pd.DataFrame(response.json())
    
    def post_insights(self, insights_data):
        """Send insights to backend"""
        response = requests.post(
            f"{self.base_url}/insights",
            headers=self.headers,
            json=insights_data
        )
        return response.json()

# Initialize client
api = APIClient(
    base_url="https://api.company.com",
    api_key=st.secrets["api"]["key"]
)

df = api.get_sales_data(
    start_date='2024-01-01',
    end_date='2024-12-31'
)
```

---

## 4. PERFORMANCE TUNING

### Caching Strategy

```python
import streamlit as st
import hashlib
import pickle

class CacheManager:
    @staticmethod
    def hash_dataframe(df):
        """Create hash of dataframe for caching"""
        return hashlib.md5(pd.util.hash_pandas_object(df, index=True).values).hexdigest()
    
    @st.cache_data(ttl=3600, max_entries=100)
    def load_data_cached(file_path):
        return pd.read_csv(file_path, encoding='latin-1')
    
    @st.cache_resource
    def get_database_connection():
        return DatabaseManager()
    
    @st.cache_data
    def process_filters(_df, filters_dict):
        """Cache filtered data"""
        filtered = _df.copy()
        for column, values in filters_dict.items():
            if values:
                filtered = filtered[filtered[column].isin(values)]
        return filtered

# Usage
df = CacheManager.load_data_cached('Sample-Superstore.csv')
filtered = CacheManager.process_filters(df, filters_dict)
```

### Query Optimization

```python
# ❌ Inefficient - loads all data
df = pd.read_csv('large_file.csv')
filtered = df[df['Category'] == 'Furniture']

# ✅ Efficient - use database query
query = "SELECT * FROM sales WHERE category = %s"
filtered = db.execute_query(query, ('Furniture',))

# ✅ Efficient - sample for display
df_display = df.sample(min(10000, len(df)))

# ✅ Efficient - use categorical dtype
df['Category'] = df['Category'].astype('category')
```

### Memory Management

```python
@st.cache_data
def load_efficient(file_path):
    """Load with memory optimization"""
    return pd.read_csv(
        file_path,
        encoding='latin-1',
        dtype={
            'Row ID': 'int32',
            'Sales': 'float32',
            'Quantity': 'int8',
            'Discount': 'float32',
            'Category': 'category',
            'Segment': 'category',
            'Region': 'category'
        },
        parse_dates=['Order Date', 'Ship Date']
    )

# Check memory usage
st.write(df.memory_usage(deep=True).sum() / 1024**2, "MB")
```

---

## 5. MULTI-USER FEATURES

### User Authentication

```python
import streamlit_authenticator as stauth
import yaml
from pathlib import Path

# Load credentials
with open('config.yaml') as file:
    config = yaml.safe_load(file)

authenticator = stauth.Authenticate(
    names=config['credentials']['usernames'].keys(),
    usernames=config['credentials']['usernames'].keys(),
    passwords=[config['credentials']['usernames'][username]['password'] 
               for username in config['credentials']['usernames']],
    cookie_name=config['cookie']['name'],
    key=config['cookie']['key'],
    cookie_expiry_days=config['cookie']['expiry_days']
)

name, authentication_status, username = authenticator.login()

if authentication_status:
    authenticator.logout()
    st.write(f'Welcome *{name}*')
    
    # Your dashboard code here
    show_dashboard(username)
    
elif authentication_status is False:
    st.error('Username/password is incorrect')
elif authentication_status is None:
    st.warning('Please enter your username and password')
```

### Role-Based Access Control (RBAC)

```python
class PermissionManager:
    ROLES = {
        'admin': ['view', 'edit', 'delete', 'export', 'manage_users'],
        'manager': ['view', 'edit', 'export'],
        'viewer': ['view']
    }
    
    @staticmethod
    def check_permission(username, action):
        """Check if user has permission"""
        user_role = get_user_role(username)  # From DB
        return action in PermissionManager.ROLES.get(user_role, [])
    
    @staticmethod
    def require_permission(action):
        """Decorator for permission check"""
        def decorator(func):
            def wrapper(*args, **kwargs):
                if PermissionManager.check_permission(st.session_state['username'], action):
                    return func(*args, **kwargs)
                else:
                    st.error(f"You don't have permission to {action}")
            return wrapper
        return decorator

# Usage
@PermissionManager.require_permission('edit')
def edit_data():
    st.write("Editing data...")

@PermissionManager.require_permission('export')
def export_data():
    st.write("Exporting data...")
```

### Audit Logging

```python
import logging
from datetime import datetime

class AuditLogger:
    def __init__(self, log_file='audit.log'):
        self.logger = logging.getLogger('audit')
        handler = logging.FileHandler(log_file)
        formatter = logging.Formatter(
            '%(asctime)s - %(username)s - %(action)s - %(details)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
    
    def log_action(self, username, action, details=''):
        """Log user action"""
        self.logger.info(
            f"action={action}",
            extra={
                'username': username,
                'action': action,
                'details': details
            }
        )

# Usage
audit = AuditLogger()
audit.log_action(
    username=st.session_state['username'],
    action='data_export',
    details=f'Exported {len(df)} records'
)
```

### User Preferences & Session Management

```python
class UserPreferences:
    @staticmethod
    def save_preferences(username, preferences):
        """Save user preferences"""
        db.execute_update(
            "UPDATE users SET preferences = %s WHERE username = %s",
            (json.dumps(preferences), username)
        )
    
    @staticmethod
    @st.cache_data
    def load_preferences(username):
        """Load user preferences"""
        result = db.execute_query(
            "SELECT preferences FROM users WHERE username = %s",
            (username,)
        )
        if result.empty:
            return {}
        return json.loads(result.iloc[0]['preferences'])
    
    @staticmethod
    def show_preferences_panel():
        """Sidebar preferences"""
        with st.sidebar.expander("⚙️ Preferences"):
            theme = st.selectbox("Theme", ["Light", "Dark", "Auto"])
            timezone = st.selectbox("Timezone", ["UTC", "EST", "IST", "GMT"])
            refresh_rate = st.slider("Auto-refresh (seconds)", 5, 3600, 60)
            
            if st.button("Save Preferences"):
                prefs = {
                    'theme': theme,
                    'timezone': timezone,
                    'refresh_rate': refresh_rate
                }
                UserPreferences.save_preferences(
                    st.session_state['username'],
                    prefs
                )
                st.success("Preferences saved!")
```

---

## 6. MONITORING & ALERTS

### Health Check Endpoint

```python
@st.cache_resource
def create_health_endpoint():
    """Create monitoring endpoint"""
    status = {
        'database': check_database(),
        'api': check_api(),
        'cache': check_cache(),
        'timestamp': pd.Timestamp.now().isoformat()
    }
    return status

def check_database():
    try:
        db.execute_query("SELECT 1")
        return 'healthy'
    except:
        return 'unhealthy'

def check_api():
    try:
        response = requests.get("https://api.example.com/health", timeout=5)
        return 'healthy' if response.status_code == 200 else 'unhealthy'
    except:
        return 'unhealthy'
```

### Error Handling & Notifications

```python
import smtplib
from email.mime.text import MIMEText

class AlertManager:
    @staticmethod
    def send_alert(subject, message, recipients):
        """Send email alert"""
        msg = MIMEText(message)
        msg['Subject'] = subject
        msg['From'] = st.secrets["email"]["sender_email"]
        msg['To'] = ', '.join(recipients)
        
        with smtplib.SMTP(st.secrets["email"]["smtp_server"], st.secrets["email"]["smtp_port"]) as server:
            server.starttls()
            server.login(
                st.secrets["email"]["sender_email"],
                st.secrets["email"]["sender_password"]
            )
            server.send_message(msg)
    
    @staticmethod
    def log_and_alert(exception, severity='error'):
        """Log error and send alert"""
        logger.error(f"{severity}: {str(exception)}")
        if severity == 'critical':
            AlertManager.send_alert(
                f"🚨 Critical Error in Dashboard",
                str(exception),
                ['ops@company.com']
            )

# Usage
try:
    # Dashboard code
    pass
except Exception as e:
    AlertManager.log_and_alert(e, severity='critical')
    st.error("An error occurred. Support team has been notified.")
```

---

## 7. DEPLOYMENT CHECKLIST

- [ ] Environment variables configured
- [ ] Database connection tested
- [ ] Authentication enabled
- [ ] SSL certificate installed
- [ ] Backup strategy in place
- [ ] Monitoring and alerting configured
- [ ] Load testing completed
- [ ] Security audit passed
- [ ] Documentation updated
- [ ] Team training completed

---

This guide provides production-ready configurations for enterprise deployment.
