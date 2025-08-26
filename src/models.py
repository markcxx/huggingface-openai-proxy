from typing import List, Optional, Union, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum


class Role(str, Enum):
    """消息角色枚举"""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    FUNCTION = "function"


class ImageUrl(BaseModel):
    """图片URL模型"""
    url: str
    detail: Optional[str] = "auto"  # "auto", "low", "high"


class TextContent(BaseModel):
    """文本内容模型"""
    type: str = "text"
    text: str


class ImageContent(BaseModel):
    """图片内容模型"""
    type: str = "image_url"
    image_url: ImageUrl


# 内容类型联合
ContentType = Union[TextContent, ImageContent]


class Message(BaseModel):
    """聊天消息模型"""
    role: Role
    content: Union[str, List[ContentType]]  # 支持字符串或多模态内容列表
    name: Optional[str] = None
    reasoning: Optional[str] = None  # 用于存储thinking内容


class ChatCompletionRequest(BaseModel):
    """聊天完成请求模型"""
    model: str
    messages: List[Message]
    temperature: Optional[float] = Field(default=1.0, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=1.0, ge=0.0, le=1.0)
    n: Optional[int] = Field(default=1, ge=1)
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    max_tokens: Optional[int] = Field(default=None, ge=1)
    presence_penalty: Optional[float] = Field(default=0.0, ge=-2.0, le=2.0)
    frequency_penalty: Optional[float] = Field(default=0.0, ge=-2.0, le=2.0)
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None


class Usage(BaseModel):
    """使用情况统计"""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class Choice(BaseModel):
    """选择项模型"""
    index: int
    message: Message
    finish_reason: Optional[str] = None


class ChatCompletionResponse(BaseModel):
    """聊天完成响应模型"""
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Choice]
    usage: Usage


class Delta(BaseModel):
    """流式响应增量"""
    role: Optional[str] = None
    content: Optional[str] = None
    reasoning: Optional[str] = None  # 用于流式thinking内容


class StreamChoice(BaseModel):
    """流式选择项"""
    index: int
    delta: Delta
    finish_reason: Optional[str] = None


class ChatCompletionStreamResponse(BaseModel):
    """流式聊天完成响应"""
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[StreamChoice]


class Model(BaseModel):
    """模型信息"""
    id: str
    object: str = "model"
    created: int
    owned_by: str


class ModelListResponse(BaseModel):
    """模型列表响应"""
    object: str = "list"
    data: List[Model]


class ErrorDetail(BaseModel):
    """错误详情"""
    message: str
    type: str
    param: Optional[str] = None
    code: Optional[str] = None


class ErrorResponse(BaseModel):
    """错误响应"""
    error: ErrorDetail