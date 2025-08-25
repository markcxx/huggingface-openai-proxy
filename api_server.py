from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from contextlib import asynccontextmanager
import logging
import uvicorn
import time
from typing import Union

from src.models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ModelListResponse,
    ErrorResponse
)
from src.converter import HuggingFaceConverter
from src.config import config

# 配置日志
logging.basicConfig(
    level=logging.INFO if not config.debug else logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 全局转换器实例
converter = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    global converter
    
    # 启动时初始化
    logger.info("Initializing Hugging Face API Proxy Server...")
    converter = HuggingFaceConverter()
    logger.info(f"Server starting on {config.host}:{config.port}")
    logger.info(f"Using Hugging Face base URL: {config.hf_base_url}")
    
    yield
    
    # 关闭时清理
    logger.info("Shutting down server...")


# 创建FastAPI应用
app = FastAPI(
    title="Hugging Face OpenAI API Proxy",
    description="A proxy server that converts OpenAI API requests to Hugging Face API calls",
    version="1.0.0",
    lifespan=lifespan
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 移除log_requests中间件以解决Vercel部署问题
# 该中间件在Vercel Serverless环境中会导致ASGI处理异常
# @app.middleware("http")
# async def log_requests(request: Request, call_next):
#     """请求日志中间件"""
#     start_time = time.time()
#     
#     # 记录请求
#     logger.info(f"{request.method} {request.url.path} - Client: {request.client.host}")
#     
#     response = await call_next(request)
#     
#     # 记录响应时间
#     process_time = time.time() - start_time
#     logger.info(f"Request completed in {process_time:.2f}s - Status: {response.status_code}")
#     
#     return response


@app.get("/")
async def root():
    """根路径"""
    return {
        "message": "Hugging Face OpenAI API Proxy",
        "version": "1.0.0",
        "endpoints": {
            "chat_completions": "/v1/chat/completions",
            "models": "/v1/models"
        }
    }


@app.get("/health")
async def health_check():
    """健康检查"""
    return {"status": "healthy", "timestamp": converter.get_current_timestamp()}


@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest, http_request: Request):
    """创建聊天完成"""
    global converter
    
    # 确保converter已初始化（Vercel环境fallback）
    if converter is None:
        logger.warning("Converter not initialized, initializing now...")
        converter = HuggingFaceConverter()
    
    # 从请求头获取客户端的API Key
    auth_header = http_request.headers.get("Authorization")
    client_api_key = None
    if auth_header and auth_header.startswith("Bearer "):
        client_api_key = auth_header[7:]  # 移除"Bearer "前缀
        logger.info(f"🔑 POST /v1/chat/completions - 使用客户端API Key: {client_api_key}")
    else:
        if config.hf_token:
            logger.warning(f"🔑 POST /v1/chat/completions - 未找到客户端API Key，使用服务端默认Key: {config.hf_token}")
        else:
            logger.warning(f"🔑 POST /v1/chat/completions - 未找到客户端API Key，且无服务端默认Key，客户端必须提供有效的HF Token")
        logger.info(f"🔑 POST /v1/chat/completions - Authorization头内容: {auth_header or 'None'}")
    
    try:
        logger.info(f"Chat completion request - Model: {request.model}, Stream: {request.stream}")
        
        if request.stream:
            # 流式响应 - 使用同步生成器以避免Vercel环境下的问题
            def generate():
                for chunk in converter.create_chat_completion_stream(request, client_api_key):
                    yield chunk
            
            return StreamingResponse(
                generate(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Content-Type-Options": "nosniff"
                }
            )
        else:
            # 非流式响应
            response = await converter.create_chat_completion(request, client_api_key)
            logger.info(f"Chat completion successful - Response ID: {response.id}")
            return response
            
    except Exception as e:
        logger.error(f"Error in chat completion: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": {
                    "message": str(e),
                    "type": "internal_error",
                    "code": "internal_error"
                }
            }
        )


@app.get("/v1/models", response_model=ModelListResponse)
async def list_models(http_request: Request):
    """获取可用模型列表"""
    global converter
    
    # 确保converter已初始化（Vercel环境fallback）
    if converter is None:
        logger.warning("Converter not initialized, initializing now...")
        converter = HuggingFaceConverter()
    
    # 从请求头获取客户端的API Key
    auth_header = http_request.headers.get("Authorization")
    client_api_key = None
    if auth_header and auth_header.startswith("Bearer "):
        client_api_key = auth_header[7:]  # 移除"Bearer "前缀
        logger.info(f"🔑 GET /v1/models - 使用客户端API Key: {client_api_key}")
    else:
        if config.hf_token:
            logger.warning(f"🔑 GET /v1/models - 未找到客户端API Key，使用服务端默认Key: {config.hf_token}")
        else:
            logger.warning(f"🔑 GET /v1/models - 未找到客户端API Key，且无服务端默认Key，客户端必须提供有效的HF Token")
    
    try:
        logger.info("Models list request")
        models = await converter.get_models(client_api_key)
        logger.info(f"Returned {len(models.data)} models")
        return models
        
    except Exception as e:
        logger.error(f"Error getting models: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": {
                    "message": str(e),
                    "type": "internal_error",
                    "code": "internal_error"
                }
            }
        )


@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    """404错误处理"""
    return JSONResponse(
        status_code=404,
        content={
            "error": {
                "message": f"Path {request.url.path} not found",
                "type": "not_found",
                "code": "not_found"
            }
        }
    )


@app.exception_handler(422)
async def validation_exception_handler(request: Request, exc):
    """请求验证错误处理"""
    return JSONResponse(
        status_code=422,
        content={
            "error": {
                "message": "Invalid request format",
                "type": "invalid_request_error",
                "code": "invalid_request_error",
                "details": str(exc)
            }
        }
    )


if __name__ == "__main__":
    import time
    
    # 启动服务器
    uvicorn.run(
        "api_server:app",
        host=config.host,
        port=config.port,
        reload=config.debug,
        log_level="info" if not config.debug else "debug"
    )