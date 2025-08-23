#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vercel API入口文件 - 符合Vercel FastAPI部署标准

这个文件是Vercel部署的标准入口点，位于api/目录下。
专门为解决Vercel环境中的ASGI兼容性问题而优化。
"""

import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入FastAPI应用
from api_server import app

# 为Vercel环境创建一个简化的ASGI处理器
class VercelASGIHandler:
    def __init__(self, app):
        self.app = app
    
    async def __call__(self, scope, receive, send):
        # 确保正确处理ASGI协议
        try:
            await self.app(scope, receive, send)
        except Exception as e:
            # 在Vercel环境中提供更好的错误处理
            await send({
                'type': 'http.response.start',
                'status': 500,
                'headers': [[b'content-type', b'application/json']],
            })
            await send({
                'type': 'http.response.body',
                'body': f'{{"error": "Internal server error: {str(e)}"}}'.encode(),
            })

# 创建Vercel兼容的应用实例
app = VercelASGIHandler(app)

# Vercel会自动识别这个app变量作为ASGI应用