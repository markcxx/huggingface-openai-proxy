#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vercel API入口文件 - 符合Vercel FastAPI部署标准

这个文件是Vercel部署的标准入口点，位于api/目录下。
"""

import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入FastAPI应用
from api_server import app

# Vercel会自动识别这个app变量作为ASGI应用
# 不需要额外的if __name__ == "__main__"代码