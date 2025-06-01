# 🚀 部署指南

本文档提供了DashScope to OpenAI API Gateway的详细部署指南。

## 📋 部署前准备

### 系统要求
- **Python**: 3.8+
- **内存**: 最少512MB，推荐1GB+
- **存储**: 最少100MB可用空间
- **网络**: 需要访问阿里云DashScope API

### 端口要求
- **默认端口**: 8000
- **可配置**: 可通过环境变量或启动参数修改

## 🏠 本地开发部署

### 1. 克隆项目
```bash
git clone <repository-url>
cd dashscope-gateway
```

### 2. 快速启动
```bash
# 使用便捷脚本（推荐）
./start.sh

# 或手动启动
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python main.py
```

### 3. 验证部署
```bash
# 检查服务状态
curl http://localhost:8000/

# 运行测试（需要DashScope API密钥）
python test_client.py
```

## 🐳 Docker部署

### 1. 单容器部署
```bash
# 构建镜像
docker build -t dashscope-gateway .

# 运行容器
docker run -d \
  --name dashscope-gateway \
  -p 8000:8000 \
  --restart unless-stopped \
  dashscope-gateway
```

### 2. Docker Compose部署
```bash
# 启动服务
docker-compose up -d

# 查看状态
docker-compose ps

# 查看日志
docker-compose logs -f
```

### 3. 自定义配置
```bash
# 自定义端口
docker run -d \
  --name dashscope-gateway \
  -p 9000:8000 \
  --restart unless-stopped \
  dashscope-gateway

# 挂载日志目录
docker run -d \
  --name dashscope-gateway \
  -p 8000:8000 \
  -v ./logs:/app/logs \
  --restart unless-stopped \
  dashscope-gateway
```

## 🖥️ 生产环境部署

### 1. systemd服务部署

#### 创建用户和目录
```bash
# 创建专用用户
sudo useradd --system --create-home --shell /bin/bash dashscope

# 创建应用目录
sudo mkdir -p /opt/dashscope-gateway
sudo chown dashscope:dashscope /opt/dashscope-gateway
```

#### 部署应用
```bash
# 切换到应用用户
sudo -u dashscope -s

# 进入应用目录
cd /opt/dashscope-gateway

# 克隆或复制代码
# git clone <repository> .
# 或者复制文件到此目录

# 创建虚拟环境
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

#### 创建systemd服务文件
```bash
sudo tee /etc/systemd/system/dashscope-gateway.service << 'EOF'
[Unit]
Description=DashScope to OpenAI API Gateway
After=network.target
Wants=network.target

[Service]
Type=simple
User=dashscope
Group=dashscope
WorkingDirectory=/opt/dashscope-gateway
Environment=PATH=/opt/dashscope-gateway/venv/bin
ExecStart=/opt/dashscope-gateway/venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=10

# 安全设置
NoNewPrivileges=yes
PrivateTmp=yes
ProtectSystem=strict
ReadWritePaths=/opt/dashscope-gateway

[Install]
WantedBy=multi-user.target
EOF
```

#### 启动服务
```bash
# 重新加载systemd配置
sudo systemctl daemon-reload

# 启用服务（开机自启）
sudo systemctl enable dashscope-gateway

# 启动服务
sudo systemctl start dashscope-gateway

# 查看状态
sudo systemctl status dashscope-gateway

# 查看日志
sudo journalctl -u dashscope-gateway -f
```

### 2. nginx反向代理

#### 安装nginx
```bash
# Ubuntu/Debian
sudo apt update && sudo apt install nginx

# CentOS/RHEL
sudo yum install nginx
# 或
sudo dnf install nginx
```

#### 配置nginx
```bash
sudo tee /etc/nginx/sites-available/dashscope-gateway << 'EOF'
server {
    listen 80;
    server_name your-domain.com;  # 替换为您的域名
    
    # 安全头
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;
    add_header X-XSS-Protection "1; mode=block";
    
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # 支持流式响应
        proxy_buffering off;
        proxy_cache off;
        proxy_read_timeout 300s;
        proxy_connect_timeout 75s;
        
        # 支持大请求
        client_max_body_size 10M;
    }
    
    # 健康检查端点
    location /health {
        proxy_pass http://127.0.0.1:8000/health;
        access_log off;
    }
}
EOF

# 启用站点
sudo ln -s /etc/nginx/sites-available/dashscope-gateway /etc/nginx/sites-enabled/

# 测试配置
sudo nginx -t

# 重启nginx
sudo systemctl restart nginx
sudo systemctl enable nginx
```

### 3. HTTPS配置（使用Let's Encrypt）

```bash
# 安装certbot
sudo apt install certbot python3-certbot-nginx

# 获取SSL证书
sudo certbot --nginx -d your-domain.com

# 自动续期测试
sudo certbot renew --dry-run
```

## 🔒 安全加固

### 1. 防火墙配置
```bash
# 使用ufw（Ubuntu）
sudo ufw allow ssh
sudo ufw allow 'Nginx Full'
sudo ufw enable

# 使用firewalld（CentOS）
sudo firewall-cmd --permanent --add-service=ssh
sudo firewall-cmd --permanent --add-service=http
sudo firewall-cmd --permanent --add-service=https
sudo firewall-cmd --reload
```

### 2. 限制访问（可选）
```nginx
# 在nginx配置中添加IP白名单
location / {
    allow 192.168.1.0/24;  # 允许内网访问
    allow 10.0.0.0/8;      # 允许VPN访问
    deny all;              # 拒绝其他IP
    
    proxy_pass http://127.0.0.1:8000;
    # ... 其他配置
}
```

### 3. 速率限制
```nginx
# 在nginx主配置中添加限流
http {
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    
    server {
        location /v1/chat/completions {
            limit_req zone=api burst=20 nodelay;
            # ... 其他配置
        }
    }
}
```

## 📊 监控配置

### 1. 基础监控
```bash
# 创建监控脚本
sudo tee /usr/local/bin/check-dashscope-gateway << 'EOF'
#!/bin/bash
HEALTH_URL="http://localhost:8000/health"
RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" $HEALTH_URL)

if [ "$RESPONSE" != "200" ]; then
    echo "DashScope Gateway is down! HTTP Code: $RESPONSE"
    # 发送告警通知
    # systemctl restart dashscope-gateway
fi
EOF

sudo chmod +x /usr/local/bin/check-dashscope-gateway

# 添加定时检查
echo "*/5 * * * * /usr/local/bin/check-dashscope-gateway" | sudo crontab -
```

### 2. 日志轮转
```bash
sudo tee /etc/logrotate.d/dashscope-gateway << 'EOF'
/opt/dashscope-gateway/logs/*.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    create 644 dashscope dashscope
    postrotate
        systemctl reload dashscope-gateway
    endscript
}
EOF
```

## 🔧 性能优化

### 1. 多worker配置
```bash
# 修改systemd服务文件
ExecStart=/opt/dashscope-gateway/venv/bin/uvicorn main:app \
  --host 0.0.0.0 \
  --port 8000 \
  --workers 4
```

### 2. 系统优化
```bash
# 增加文件描述符限制
echo "dashscope soft nofile 65536" | sudo tee -a /etc/security/limits.conf
echo "dashscope hard nofile 65536" | sudo tee -a /etc/security/limits.conf

# 优化内核参数
echo "net.core.somaxconn = 1024" | sudo tee -a /etc/sysctl.conf
echo "net.core.netdev_max_backlog = 5000" | sudo tee -a /etc/sysctl.conf
sudo sysctl -p
```

## 🚨 故障排除

### 1. 常见问题

#### 服务无法启动
```bash
# 检查端口占用
sudo netstat -tlnp | grep :8000

# 检查权限
sudo ls -la /opt/dashscope-gateway/

# 检查Python环境
sudo -u dashscope /opt/dashscope-gateway/venv/bin/python --version
```

#### 依赖安装失败
```bash
# 更新pip
sudo -u dashscope /opt/dashscope-gateway/venv/bin/pip install --upgrade pip

# 清理缓存
sudo -u dashscope /opt/dashscope-gateway/venv/bin/pip cache purge

# 手动安装依赖
sudo -u dashscope /opt/dashscope-gateway/venv/bin/pip install -r requirements.txt --no-cache-dir
```

### 2. 日志分析
```bash
# 查看服务日志
sudo journalctl -u dashscope-gateway -f

# 查看nginx日志
sudo tail -f /var/log/nginx/access.log
sudo tail -f /var/log/nginx/error.log

# 搜索错误
sudo journalctl -u dashscope-gateway | grep ERROR
```

### 3. 性能分析
```bash
# 查看系统资源
htop

# 查看网络连接
sudo ss -tlnp | grep :8000

# 检查服务响应时间
curl -w "@-" -s -o /dev/null http://localhost:8000/health << 'EOF'
响应时间: %{time_total}s
HTTP状态: %{http_code}
DNS解析: %{time_namelookup}s
连接时间: %{time_connect}s
EOF
```

## 📈 扩展部署

### 1. 负载均衡
```nginx
upstream dashscope_backend {
    server 127.0.0.1:8000;
    server 127.0.0.1:8001;
    server 127.0.0.1:8002;
}

server {
    location / {
        proxy_pass http://dashscope_backend;
        # ... 其他配置
    }
}
```

### 2. 容器编排（Docker Swarm）
```yaml
version: '3.8'
services:
  dashscope-gateway:
    image: dashscope-gateway:latest
    deploy:
      replicas: 3
      restart_policy:
        condition: on-failure
    ports:
      - "8000:8000"
    networks:
      - dashscope-net

networks:
  dashscope-net:
    driver: overlay
```

### 3. Kubernetes部署
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: dashscope-gateway
spec:
  replicas: 3
  selector:
    matchLabels:
      app: dashscope-gateway
  template:
    metadata:
      labels:
        app: dashscope-gateway
    spec:
      containers:
      - name: dashscope-gateway
        image: dashscope-gateway:latest
        ports:
        - containerPort: 8000
---
apiVersion: v1
kind: Service
metadata:
  name: dashscope-gateway-service
spec:
  selector:
    app: dashscope-gateway
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

## ✅ 部署检查清单

- [ ] 系统依赖已安装
- [ ] 应用代码已部署
- [ ] Python虚拟环境已创建
- [ ] 依赖包已安装
- [ ] systemd服务已配置
- [ ] nginx反向代理已配置
- [ ] 防火墙规则已设置
- [ ] HTTPS证书已配置
- [ ] 监控脚本已部署
- [ ] 日志轮转已配置
- [ ] 服务自启动已启用
- [ ] 健康检查通过
- [ ] 功能测试通过

## 📞 支持

如果在部署过程中遇到问题，请：
1. 检查日志文件
2. 查阅故障排除章节
3. 提交Issue到项目仓库 