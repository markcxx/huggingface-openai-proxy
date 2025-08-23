#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vercel入口文件 - 专门为Vercel部署优化的应用入口

这个文件是Vercel部署的入口点，确保FastAPI应用能在Serverless环境中正确运行。
"""

from api_server import app

# Vercel需要一个名为app的变量作为ASGI应用入口
# 这里直接导出FastAPI应用实例
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)