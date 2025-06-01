# DashScope to OpenAI API Gateway 🚀

**公共服务** - 将阿里百炼DashScope API转换为OpenAI API格式的中间件网关，让您可以使用OpenAI SDK调用阿里百炼的farui-plus模型。

> ⚠️ **重要说明**: 这是一个公共服务，不需要预配置API密钥。用户在每次请求时通过Authorization header传递自己的DashScope API密钥。

## 🌟 功能特性

- ✅ **完全兼容OpenAI API格式** - 支持所有OpenAI SDK调用方式
- ✅ **公共服务设计** - 用户动态传递API密钥，无需服务端配置
- ✅ **流式和非流式响应** - 完整支持实时流式输出
- ✅ **完整参数映射** - temperature、max_tokens、top_p、stop等参数全支持
- ✅ **安全日志** - 详细日志记录，API密钥经过哈希处理保护隐私
- ✅ **CORS支持** - 支持跨域请求
- ✅ **健康检查** - 内置监控端点

## 📡 支持的API端点

- `GET /` - 服务信息和使用说明
- `GET /health` - 健康检查
- `GET /v1/models` - 获取可用模型列表
- `POST /v1/chat/completions` - 聊天完成（完全兼容OpenAI格式）

## 🚀 快速开始

### 本地运行

1. **安装依赖**
```bash
pip install -r requirements.txt
```

2. **启动服务**
```bash
python main.py
```

或使用便捷脚本：
```bash
./start.sh
```

服务将在 `http://localhost:8000` 启动。

## 💡 使用方法

### 🔑 重要：API密钥说明

- **无需预配置**: 服务不需要预先配置任何API密钥
- **用户提供**: 每个用户在请求时提供自己的DashScope API密钥
- **安全传递**: 通过`Authorization: Bearer YOUR_DASHSCOPE_API_KEY`传递
- **隐私保护**: 服务不存储任何用户的API密钥

### 1. 使用OpenAI Python SDK

```python
import openai

# 初始化客户端，指向网关地址
client = openai.OpenAI(
    api_key="您的DashScope-API-Key",  # 🔑 使用您自己的密钥
    base_url="http://localhost:8000/v1"  # 🌐 网关地址
)

# 非流式调用
response = client.chat.completions.create(
    model="farui-plus",
    messages=[
        {"role": "system", "content": "你是一个有用的AI助手。"},
        {"role": "user", "content": "你好！"}
    ],
    temperature=0.7,
    max_tokens=100
)

print(response.choices[0].message.content)

# 流式调用
stream = client.chat.completions.create(
    model="farui-plus",
    messages=[
        {"role": "user", "content": "写一首短诗"}
    ],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end='')
```

### 2. 使用curl

**非流式请求：**
```bash
curl -X POST "http://localhost:8000/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer 您的DashScope-API-Key" \
  -d '{
    "model": "farui-plus",
    "messages": [
      {"role": "system", "content": "你是一个有用的AI助手。"},
      {"role": "user", "content": "你好！"}
    ],
    "temperature": 0.7,
    "max_tokens": 100
  }'
```

**流式请求：**
```bash
curl -X POST "http://localhost:8000/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer 您的DashScope-API-Key" \
  -d '{
    "model": "farui-plus",
    "messages": [
      {"role": "user", "content": "写一首短诗"}
    ],
    "stream": true
  }'
```

### 3. 测试客户端

运行内置的测试客户端：

```bash
# 设置环境变量（可选）
export DASHSCOPE_API_KEY=your_api_key

# 运行测试
python test_client.py
```

测试客户端会：
- 自动检查服务状态
- 提示输入API密钥（如果未设置环境变量）
- 运行完整的功能测试
- 显示使用示例

### 4. 兼容性

任何支持自定义OpenAI API端点的工具都可以使用：

| 工具/SDK | 配置方法 |
|---------|---------|
| **OpenAI Python SDK** | `openai.OpenAI(api_key="your_key", base_url="http://your-server:8000/v1")` |
| **LangChain** | `ChatOpenAI(openai_api_key="your_key", openai_api_base="http://your-server:8000/v1")` |
| **LlamaIndex** | 设置`openai.api_base = "http://your-server:8000/v1"` |
| **Cursor IDE** | 在设置中配置自定义API端点 |
| **Continue.dev** | 在配置文件中设置endpoint |

## 🔧 API参数映射

| OpenAI参数 | DashScope参数 | 说明 |
|-----------|--------------|------|
| model | model | 固定为"farui-plus" |
| messages | messages | 消息格式完全兼容 |
| temperature | temperature | 温度参数 (0.0-2.0) |
| max_tokens | max_tokens | 最大输出token数 |
| top_p | top_p | 核采样参数 (0.0-1.0) |
| stream | stream | 是否启用流式输出 |
| stop | stop | 停止词列表 |

## 🛡️ 安全特性

### API密钥保护
- ✅ **不存储密钥**: 服务不保存任何用户的API密钥
- ✅ **哈希日志**: 日志中仅记录API密钥的MD5哈希前8位
- ✅ **请求隔离**: 每个请求独立处理，密钥不会泄露给其他用户

### 日志示例
```
2024-01-01 12:00:00 - Request started - IP: 192.168.1.100, Method: POST, Path: /v1/chat/completions, API Key Hash: a1b2c3d4
2024-01-01 12:00:02 - Request completed - IP: 192.168.1.100, Status: 200, Time: 2.34s, API Key Hash: a1b2c3d4
```

## 🐳 Docker部署

### 环境变量配置

| 环境变量 | 说明 | 默认值 |
|---------|------|--------|
| `DASHSCOPE_BASE_URL` | DashScope API请求地址 | `https://dashscope.aliyuncs.com` |

### 构建和运行

```bash
# 构建镜像
docker build -t dashscope-gateway .

# 运行容器（使用默认地址）
docker run -p 8000:8000 dashscope-gateway

# 运行容器（使用自定义地址）
docker run -p 8000:8000 -e DASHSCOPE_BASE_URL=https://your-custom-endpoint.com dashscope-gateway
```

### Docker Compose

```bash
docker-compose up -d
```

## 🖥️ 服务器部署

### 1. 使用systemd服务

创建服务文件 `/etc/systemd/system/dashscope-gateway.service`：

```ini
[Unit]
Description=DashScope to OpenAI API Gateway
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory=/path/to/your/app
Environment=PATH=/path/to/your/venv/bin
ExecStart=/path/to/your/venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000
Restart=always

[Install]
WantedBy=multi-user.target
```

启动服务：
```bash
sudo systemctl enable dashscope-gateway
sudo systemctl start dashscope-gateway
```

### 2. 使用nginx反向代理

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # 支持流式响应
        proxy_buffering off;
        proxy_cache off;
    }
}
```

## 📊 监控和日志

### 查看日志
```bash
# Docker
docker logs dashscope-gateway

# systemd
journalctl -u dashscope-gateway -f

# 直接运行
# 日志会输出到控制台
```

### 健康检查
```bash
curl http://localhost:8000/health
```

返回：
```json
{
  "status": "healthy",
  "timestamp": 1704067200,
  "service": "DashScope to OpenAI API Gateway"
}
```

## 🚨 错误处理

服务返回标准OpenAI格式的错误响应：

```json
{
  "error": {
    "message": "错误描述",
    "type": "错误类型",
    "code": "错误代码"
  }
}
```

### 常见错误类型

| 错误类型 | 说明 | 解决方案 |
|---------|------|---------|
| `authentication_error` | API密钥无效 | 检查DashScope API密钥是否正确 |
| `dashscope_error` | DashScope API错误 | 检查API调用参数和配额 |
| `internal_error` | 服务内部错误 | 查看服务日志 |

## 🔧 故障排除

### 1. 连接问题
```bash
# 检查服务状态
curl http://localhost:8000/

# 预期返回服务信息
```

### 2. API密钥问题
- 确认DashScope API密钥有效
- 检查密钥权限和配额
- 确认密钥格式正确（通常以"sk-"开头）

### 3. 模型访问问题
- 确认账户有farui-plus模型访问权限
- 检查阿里云百炼控制台的模型状态

### 4. 调试模式
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --log-level debug
```

## 🎯 使用场景

### 适用场景
- 🔄 **API格式迁移**: 将现有OpenAI应用迁移到阿里百炼
- 🏢 **企业服务**: 为团队提供统一的API接口
- 🔧 **开发测试**: 在不同模型间快速切换测试
- 📚 **教学演示**: 用于AI开发教学和演示

### 典型用户
- AI应用开发者
- 企业技术团队
- 教育工作者
- 研究人员

## 🤝 贡献

欢迎提交Issue和Pull Request！

## 📄 许可证

MIT License

## 📋 更新日志

### v1.0.0 (Current)
- ✅ 完整的OpenAI API兼容性
- ✅ 公共服务架构
- ✅ 安全的API密钥处理
- ✅ 详细的日志和监控
- ✅ 完善的错误处理
- ✅ 流式响应支持
- ✅ CORS支持
- ✅ Docker部署支持 