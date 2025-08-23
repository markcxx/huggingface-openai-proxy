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

import sys
import os

# 添加父目录到Python路径，以便导入模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api_server import app

# 直接导出app对象供Vercel使用
# Vercel会直接使用这个app对象，而不是通过uvicorn.run()
app = app

# Vercel函数入口点
# 只需要导出app对象，不需要启动逻辑