# Hugging Face OpenAI API 代理服务

这是一个基于 FastAPI 的代理服务，将 OpenAI 格式的 API 请求转换为 Hugging Face API 调用，让您可以无缝使用 Hugging Face 的模型服务。

## 🚀 功能特性

- ✅ **OpenAI 兼容 API**：完全兼容 OpenAI API 格式
- ✅ **流式响应支持**：支持 Server-Sent Events (SSE) 流式输出
- ✅ **多模型支持**：支持 Hugging Face 上的各种模型
- ✅ **CORS 支持**：支持跨域访问
- ✅ **错误处理**：完善的错误处理和日志记录
- ✅ **高性能**：基于 FastAPI 和异步处理

## 📋 API 端点

- `POST /v1/chat/completions` - 聊天完成（支持流式和非流式）
- `GET /v1/models` - 获取可用模型列表
- `GET /health` - 健康检查
- `GET /` - 服务信息

## 🛠️ 安装和配置

### 方法一：使用 Conda（推荐）

1. **创建 conda 虚拟环境**：
```bash
conda env create -f environment.yml
conda activate huggingface
```

### 方法二：使用 pip

1. **创建虚拟环境**：
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

2. **安装依赖**：
```bash
pip install -r requirements.txt
```

### 配置环境变量

创建 `.env` 文件并配置以下变量：

```env
# 必需配置
HF_TOKEN=your_huggingface_token_here

# 可选配置
HF_BASE_URL=https://router.huggingface.co/v1
HOST=0.0.0.0
PORT=8000
DEBUG=false
API_PREFIX=/v1
CORS_ORIGINS=*
DEFAULT_MODEL=openai/gpt-oss-120b:fireworks-ai
REQUEST_TIMEOUT=300
```

**获取 Hugging Face Token**：
1. 访问 [Hugging Face Settings](https://huggingface.co/settings/tokens)
2. 创建一个新的 Access Token
3. 将 token 添加到 `.env` 文件中

## 🚀 启动服务

### 方法一：使用主启动脚本（推荐）
```bash
python main.py
```

### 方法二：直接使用uvicorn
```bash
uvicorn api_server:app --host 0.0.0.0 --port 8000
```

### 方法三：安装后使用命令行
```bash
# 安装项目
pip install -e .

# 使用命令行启动
hf-openai-proxy
```

### 使用 Docker（可选）
```bash
# 构建镜像
docker build -t hf-openai-proxy .

# 运行容器
docker run -p 8000:8000 --env-file .env hf-openai-proxy
```

## 📖 使用示例

### 1. 非流式请求

```python
import requests

url = "http://localhost:8000/v1/chat/completions"
headers = {"Content-Type": "application/json"}
data = {
    "model": "openai/gpt-oss-120b:fireworks-ai",
    "messages": [
        {"role": "user", "content": "你好，请介绍一下你自己"}
    ],
    "temperature": 0.7,
    "max_tokens": 1000
}

response = requests.post(url, json=data, headers=headers)
print(response.json())
```

### 2. 流式请求

```python
import requests
import json

url = "http://localhost:8000/v1/chat/completions"
headers = {"Content-Type": "application/json"}
data = {
    "model": "openai/gpt-oss-120b:fireworks-ai",
    "messages": [
        {"role": "user", "content": "请写一首关于春天的诗"}
    ],
    "stream": True,
    "temperature": 0.7
}

response = requests.post(url, json=data, headers=headers, stream=True)

for line in response.iter_lines():
    if line:
        line = line.decode('utf-8')
        if line.startswith('data: '):
            data_str = line[6:]  # 移除 'data: ' 前缀
            if data_str.strip() == '[DONE]':
                break
            try:
                data = json.loads(data_str)
                if 'choices' in data and data['choices']:
                    content = data['choices'][0]['delta'].get('content', '')
                    if content:
                        print(content, end='', flush=True)
            except json.JSONDecodeError:
                continue
```

### 3. 使用 OpenAI 客户端

```python
from openai import OpenAI

# 配置客户端指向本地代理服务
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy-key"  # 代理服务不需要真实的 OpenAI key
)

# 非流式调用
response = client.chat.completions.create(
    model="openai/gpt-oss-120b:fireworks-ai",
    messages=[
        {"role": "user", "content": "你好，世界！"}
    ]
)
print(response.choices[0].message.content)

# 流式调用
stream = client.chat.completions.create(
    model="openai/gpt-oss-120b:fireworks-ai",
    messages=[
        {"role": "user", "content": "请详细介绍一下人工智能"}
    ],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

### 4. 获取模型列表

```python
import requests

response = requests.get("http://localhost:8000/v1/models")
models = response.json()
print(f"可用模型数量: {len(models['data'])}")
for model in models['data']:
    print(f"- {model['id']}")
```

## 🔧 配置说明

| 环境变量 | 默认值 | 说明 |
|---------|--------|------|
| `HF_TOKEN` | 无 | Hugging Face API Token（必需） |
| `HF_BASE_URL` | `https://router.huggingface.co/v1` | Hugging Face API 基础 URL |
| `HOST` | `0.0.0.0` | 服务器监听地址 |
| `PORT` | `8000` | 服务器端口 |
| `DEBUG` | `false` | 是否启用调试模式 |
| `CORS_ORIGINS` | `*` | 允许的跨域来源 |
| `DEFAULT_MODEL` | `openai/gpt-oss-120b:fireworks-ai` | 默认模型 |
| `REQUEST_TIMEOUT` | `300` | 请求超时时间（秒） |

## 📁 项目结构

```
Huggingface-api/
├── src/                   # 核心源代码包
│   ├── __init__.py        # 包初始化文件
│   ├── models.py          # Pydantic 数据模型
│   ├── converter.py       # 请求/响应转换器
│   └── config.py          # 配置管理
├── api_server.py          # FastAPI 应用定义
├── main.py                # 主启动脚本
├── setup.py               # 包安装配置
├── requirements.txt       # Python 依赖
├── environment.yml        # Conda 环境配置
├── .env                   # 环境变量（需要创建）
├── .env.example           # 环境变量示例
├── .gitignore             # Git 忽略文件
└── README.md              # 说明文档
```

## 🧪 测试

运行测试脚本：
```bash
python test_client.py
```

## 🐛 故障排除

### 常见问题

1. **HF_TOKEN 错误**
   - 确保在 `.env` 文件中正确设置了 `HF_TOKEN`
   - 验证 token 是否有效且有足够权限

2. **模型不可用**
   - 检查模型名称是否正确
   - 确认您的 Hugging Face 账户可以访问该模型

3. **连接超时**
   - 检查网络连接
   - 增加 `REQUEST_TIMEOUT` 值

4. **CORS 错误**
   - 在 `.env` 中正确配置 `CORS_ORIGINS`

### 日志查看

服务器会输出详细的日志信息，包括：
- 请求信息
- 响应时间
- 错误详情

## 📄 许可证

MIT License

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📞 支持

如果您遇到问题，请：
1. 查看日志输出
2. 检查配置是否正确
3. 提交 Issue 描述问题