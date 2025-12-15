
# Enterprise Superstore Dashboard - Deployment Guide

## Overview
This guide covers deploying the Streamlit dashboard as an enterprise solution with multiple hosting and architecture options.

---

## 1. LOCAL DEVELOPMENT

### Prerequisites
```bash
# Install required packages
pip install streamlit pandas numpy plotly
```

### Running Locally
```bash
# Navigate to project directory
cd /path/to/dashboard

# Run the dashboard
streamlit run streamlit-dashboard.py

# Access at: http://localhost:8501
```

---

## 2. DOCKER CONTAINERIZATION (Recommended for Enterprise)

### Docker Setup
Create a `Dockerfile` in your project root:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY streamlit-dashboard.py .
COPY Sample-Superstore.csv .

# Expose port
EXPOSE 8501

# Configure Streamlit
RUN mkdir -p ~/.streamlit && \
    echo "[server]" > ~/.streamlit/config.toml && \
    echo "port = 8501" >> ~/.streamlit/config.toml && \
    echo "headless = true" >> ~/.streamlit/config.toml && \
    echo "enableCORS = false" >> ~/.streamlit/config.toml

# Run application
CMD ["streamlit", "run", "streamlit-dashboard.py", "--server.port=8501"]
```

### Create requirements.txt
```txt
streamlit==1.28.1
pandas==2.1.1
numpy==1.24.3
plotly==5.17.0
```

### Build and Run Docker Container
```bash
# Build image
docker build -t superstore-dashboard:latest .

# Run container
docker run -p 8501:8501 \
  -v /path/to/data:/app/data \
  superstore-dashboard:latest

# Access at: http://localhost:8501
```

### Docker Compose (Multi-service Setup)
Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  streamlit:
    build: .
    ports:
      - "8501:8501"
    volumes:
      - ./Sample-Superstore.csv:/app/Sample-Superstore.csv
      - ./data:/app/data
    environment:
      - STREAMLIT_SERVER_HEADLESS=true
      - STREAMLIT_SERVER_ENABLECORS=false
    restart: unless-stopped

  # Optional: PostgreSQL for data caching
  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: superstore
      POSTGRES_USER: admin
      POSTGRES_PASSWORD: secure_password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

volumes:
  postgres_data:
```

Run with: `docker-compose up -d`

---

## 3. ENTERPRISE HOSTING OPTIONS

### 3.1 Streamlit Cloud (Easiest)
**Best for:** Teams, quick deployment, GitHub integration

#### Setup Steps:
1. Push code to GitHub repository
2. Go to https://streamlit.io/cloud
3. Click "New app" → Connect to GitHub
4. Select repository and main file
5. Deploy with one click

**Advantages:**
- ✅ Free tier available
- ✅ One-click deployment from GitHub
- ✅ Automatic updates
- ✅ Built-in authentication optional
- ✅ Custom domain support (paid)

**Limitations:**
- Community sharing by default
- Resource limits on free tier
- Limited to public/private GitHub access

**Cost:** Free tier or ~$7-99/month for upgraded tiers

---

### 3.2 AWS EC2 + Streamlit (Enterprise Grade)

#### Architecture:
```
User → ALB → EC2 (Streamlit) → RDS (PostgreSQL)
     ↓
   Route53
```

#### Deployment Steps:

**Step 1: Launch EC2 Instance**
```bash
# Use Ubuntu 22.04 LTS
# Instance type: t3.medium or higher (for production)
# Security group: Allow inbound 22 (SSH), 80 (HTTP), 443 (HTTPS)
```

**Step 2: Install on EC2**
```bash
# SSH into instance
ssh -i your-key.pem ubuntu@your-instance-ip

# Update system
sudo apt update && sudo apt upgrade -y

# Install Python and dependencies
sudo apt install python3.11 python3-pip nginx supervisor -y

# Clone repository
git clone https://github.com/your-org/superstore-dashboard.git
cd superstore-dashboard

# Install Python packages
pip3 install -r requirements.txt
```

**Step 3: Configure Supervisor (Process Management)**
Create `/etc/supervisor/conf.d/streamlit.conf`:

```ini
[program:streamlit]
command=/usr/bin/python3 -m streamlit run /home/ubuntu/superstore-dashboard/streamlit-dashboard.py \
    --server.port 8501 \
    --server.headless true \
    --logger.level=info
directory=/home/ubuntu/superstore-dashboard
user=ubuntu
autostart=true
autorestart=true
redirect_stderr=true
stdout_logfile=/var/log/streamlit/stdout.log
```

**Step 4: Configure Nginx Reverse Proxy**
Create `/etc/nginx/sites-available/dashboard`:

```nginx
upstream streamlit {
    server 127.0.0.1:8501;
}

server {
    listen 80;
    server_name your-domain.com;

    client_max_body_size 100M;

    location / {
        proxy_pass http://streamlit;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_buffering off;
    }
}
```

Enable and restart:
```bash
sudo ln -s /etc/nginx/sites-available/dashboard /etc/nginx/sites-enabled/
sudo systemctl restart nginx supervisor
```

**Step 5: SSL Certificate (Let's Encrypt)**
```bash
sudo apt install certbot python3-certbot-nginx -y
sudo certbot --nginx -d your-domain.com
```

**AWS Cost Estimate:**
- EC2 (t3.medium): ~$30/month
- Data transfer: ~$5-15/month
- **Total: ~$35-45/month**

---

### 3.3 Google Cloud Run (Serverless)

**Best for:** Scalable, event-driven, minimal ops

#### Deployment:
```bash
# Install Google Cloud SDK
# Authenticate
gcloud auth login

# Set project
gcloud config set project your-project-id

# Create .gcloudignore file
echo ".git" > .gcloudignore
echo ".gitignore" >> .gcloudignore

# Deploy
gcloud run deploy superstore-dashboard \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --timeout 3600

# Access the provided URL
```

**Cost:** Pay-per-invocation (~$0.0000002 per request, ~$3-5/month for typical usage)

---

### 3.4 Azure App Service (Enterprise)

**Best for:** Azure ecosystem, enterprise integration

#### Steps:
```bash
# Install Azure CLI
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash

# Login
az login

# Create resource group
az group create --name superstore-rg --location eastus

# Create App Service Plan
az appservice plan create --name superstore-plan \
  --resource-group superstore-rg \
  --sku B2

# Create Web App
az webapp create --resource-group superstore-rg \
  --plan superstore-plan \
  --name superstore-dashboard

# Deploy from GitHub
az webapp deployment github-actions add \
  --resource-group superstore-rg \
  --name superstore-dashboard \
  --repo your-github-repo \
  --branch main
```

**Cost:** ~$50-100/month for B2 tier

---

## 4. ENTERPRISE ENHANCEMENTS

### 4.1 Authentication & Authorization
Add to dashboard for enterprise users:

```python
import streamlit as st
from streamlit_authenticator import Authenticate

# Create authenticator
authenticator = Authenticate(
    names=['admin', 'user1'],
    usernames=['admin', 'user1'],
    passwords=['hashed_password1', 'hashed_password2'],
    cookie_name='dashboard_auth',
    key='dashboard_key',
    cookie_expiry_days=30
)

# Login widget
authenticator.login()

if st.session_state['authentication_status']:
    st.write(f'Welcome *{st.session_state["name"]}*')
    # Rest of dashboard code
elif st.session_state['authentication_status'] is False:
    st.error('Username/password is incorrect')
else:
    st.warning('Please enter your username and password')
```

### 4.2 Database Connection (PostgreSQL)
```python
import psycopg2
from psycopg2.pool import SimpleConnectionPool

@st.cache_resource
def init_connection():
    return psycopg2.connect(
        host="your-db-host.rds.amazonaws.com",
        database="superstore",
        user="admin",
        password="your-password"
    )

conn = init_connection()

# Load data from database
def load_data_from_db():
    query = "SELECT * FROM sales_data"
    return pd.read_sql(query, conn)

df = load_data_from_db()
```

### 4.3 Monitoring & Logging
```python
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/var/log/streamlit/dashboard.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

# Log important events
logger.info(f"Dashboard loaded with {len(df)} records")
logger.info(f"User filtered data: {len(filtered_df)} records")
```

### 4.4 Performance Optimization
```python
# Use session state for caching
if 'filtered_data' not in st.session_state:
    st.session_state.filtered_data = apply_filters(df)

# Lazy load charts
if st.session_state.show_detailed_charts:
    st.plotly_chart(...)

# Reduce data processing
df = df.sample(n=min(10000, len(df)))  # Sample for large datasets
```

---

## 5. CI/CD PIPELINE (GitHub Actions)

Create `.github/workflows/deploy.yml`:

```yaml
name: Deploy to AWS

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Configure AWS credentials
        uses: aws-actions/configure-aws-credentials@v1
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: us-east-1

      - name: Build and push Docker image
        run: |
          aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin $ECR_REGISTRY
          docker build -t $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG .
          docker push $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG

      - name: Deploy to EC2
        run: |
          # SSH and pull latest changes
          ssh -i ${{ secrets.EC2_KEY }} ubuntu@${{ secrets.EC2_HOST }} \
            "cd superstore-dashboard && git pull origin main && \
             docker pull $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG && \
             docker-compose up -d"
```

---

## 6. SECURITY BEST PRACTICES

### Environment Variables
```bash
# Create .env file (never commit)
DATABASE_URL=postgresql://user:password@host:5432/db
STREAMLIT_LOGGER_LEVEL=info
MAX_UPLOAD_SIZE=104857600
```

### SSL/TLS
- Always use HTTPS in production
- Implement certificate rotation
- Use strong cipher suites

### Access Control
- Implement role-based access control (RBAC)
- Enable audit logging
- Use VPN or IP whitelisting for on-premises access

---

## 7. MONITORING & MAINTENANCE

### Health Check
```bash
# Add to cron for periodic checks
*/5 * * * * curl -f http://localhost:8501/_stcore/health || systemctl restart streamlit
```

### Backup Strategy
```bash
# Daily backup to S3
0 2 * * * /usr/local/bin/backup-dashboard.sh
```

### Log Monitoring
```bash
# Track errors
tail -f /var/log/streamlit/stdout.log | grep ERROR
```

---

## 8. RECOMMENDED PRODUCTION ARCHITECTURE

```
┌─────────────────┐
│   Users (Web)   │
└────────┬────────┘
         │
    ┌────▼─────────┐
    │  CloudFlare  │ (DDoS Protection, Cache)
    └────┬─────────┘
         │
    ┌────▼──────────────┐
    │  Load Balancer    │ (AWS ALB / Azure LB)
    └────┬──────────────┘
         │
    ┌────┴────┬────────┐
    │          │        │
┌───▼──┐  ┌──▼──┐  ┌──▼──┐
│ App1 │  │ App2 │  │ App3 │ (Streamlit - Auto-scaling)
└───┬──┘  └──┬──┘  └──┬──┘
    │        │        │
    └────┬───┴────┬───┘
         │        │
    ┌────▼──┐  ┌──▼──────────┐
    │  Cache │  │ PostgreSQL  │ (Multi-AZ)
    │ (Redis)│  │ RDS         │
    └────────┘  └─────────────┘
```

**Features:**
- Auto-scaling based on load
- Multi-AZ deployment for HA
- Automated backups
- Read replicas for reporting
- Cache layer for frequently accessed data

---

## 9. COST COMPARISON

| Option | Startup | Monthly | Scalability |
|--------|---------|---------|------------|
| Streamlit Cloud | $0 | $0-99 | Good |
| Docker (Self-hosted) | $100 | $35-50 | Limited |
| AWS EC2 | $100 | $35-100 | Excellent |
| AWS ECS (Containers) | $200 | $50-150 | Excellent |
| Google Cloud Run | $50 | $5-20 | Excellent |
| Azure App Service | $100 | $50-100 | Good |
| Kubernetes (Self-hosted) | $500 | $100-300 | Excellent |

---

## 10. NEXT STEPS

1. **Choose Hosting:** Streamlit Cloud (fastest) or AWS (most control)
2. **Add Authentication:** Protect dashboard with login
3. **Implement Database:** Move from CSV to PostgreSQL for real-time updates
4. **Setup Monitoring:** Add alerts and logging
5. **Create CI/CD:** Automate deployments
6. **Security Audit:** Implement SSL, backups, access control

---

## 11. SUPPORT & RESOURCES

- **Streamlit Docs:** https://docs.streamlit.io
- **Plotly Docs:** https://plotly.com/python
- **AWS Documentation:** https://docs.aws.amazon.com
- **Docker Documentation:** https://docs.docker.com
- **Streamlit Community Forum:** https://discuss.streamlit.io

---

For questions or issues, refer to the documentation or deployment logs.
