import os
from typing import Optional
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()


class Config:
    """应用配置类"""
    
    def __init__(self):
        # Hugging Face 配置
        self.hf_token: str = os.getenv("HF_TOKEN", "")
        self.hf_base_url: str = os.getenv("HF_BASE_URL", "https://router.huggingface.co/v1")
        
        # 服务器配置
        self.host: str = os.getenv("HOST", "0.0.0.0")
        self.port: int = int(os.getenv("PORT", "8000"))
        self.debug: bool = os.getenv("DEBUG", "false").lower() == "true"
        
        # API配置
        self.api_prefix: str = os.getenv("API_PREFIX", "/v1")
        self.cors_origins: list = os.getenv("CORS_ORIGINS", "*").split(",")
        
        # 默认模型配置
        self.default_model: str = os.getenv("DEFAULT_MODEL", "openai/gpt-oss-120b:fireworks-ai")
        
        # 请求超时配置
        self.request_timeout: int = int(os.getenv("REQUEST_TIMEOUT", "300"))
        
        # 验证必要的配置
        self._validate_config()
    
    def _validate_config(self):
        """验证配置"""
        if not self.hf_token:
            raise ValueError("HF_TOKEN environment variable is required")
    
    @property
    def is_development(self) -> bool:
        """是否为开发环境"""
        return self.debug


# 全局配置实例
config = Config()