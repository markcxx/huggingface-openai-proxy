#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hugging Face OpenAI API Proxy - 主启动脚本

这是一个FastAPI代理服务，将OpenAI格式的API请求转换为Hugging Face API调用。
支持reasoning模型的thinking内容处理。

使用方法:
    python main.py

或者使用uvicorn:
    uvicorn main:app --host 0.0.0.0 --port 8000
"""

import uvicorn
from api_server import app
from src.config import config


def main():
    """主函数 - 启动FastAPI服务器"""
    print("🚀 启动 Hugging Face OpenAI API 代理服务...")
    print(f"📍 服务地址: http://{config.host}:{config.port}")
    print(f"📖 API文档: http://{config.host}:{config.port}/docs")
    print(f"🔧 配置模式: {'开发' if config.debug else '生产'}")
    
    uvicorn.run(
        "api_server:app",
        host=config.host,
        port=config.port,
        reload=config.debug,
        log_level="info" if not config.debug else "debug"
    )


if __name__ == "__main__":
    main()