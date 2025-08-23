import time
import uuid
import re
from typing import List, Dict, Any, AsyncGenerator
from openai import OpenAI
from .models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionStreamResponse,
    Choice,
    StreamChoice,
    Delta,
    Usage,
    Message,
    Model,
    ModelListResponse
)
from .config import config
import json
import logging

logger = logging.getLogger(__name__)


class HuggingFaceConverter:
    """Hugging Face API转换器"""
    
    def __init__(self):
        self._client = None
    
    @property
    def client(self):
        """懒加载OpenAI客户端"""
        if self._client is None:
            # 如果没有配置HF_TOKEN，使用默认值，让客户端自行配置
            api_key = config.hf_token or "dummy-key"
            self._client = OpenAI(
                base_url=config.hf_base_url,
                api_key=api_key,
            )
        return self._client
    
    def convert_messages_to_hf_format(self, messages: List[Message]) -> List[Dict[str, str]]:
        """将OpenAI消息格式转换为Hugging Face格式"""
        hf_messages = []
        for msg in messages:
            hf_messages.append({
                "role": msg.role.value,
                "content": msg.content
            })
        return hf_messages
    
    def generate_response_id(self) -> str:
        """生成响应ID"""
        return f"chatcmpl-{uuid.uuid4().hex[:29]}"
    
    def get_current_timestamp(self) -> int:
        """获取当前时间戳"""
        return int(time.time())
    
    def estimate_tokens(self, text: str) -> int:
        """估算token数量（简单实现）"""
        if text is None:
            return 0
        # 简单的token估算，实际应该使用tokenizer
        return len(text.split()) + len(text) // 4
    
    def parse_thinking_content(self, content: str) -> tuple[str, str]:
        """解析thinking内容，分离思考过程和最终回答"""
        if content is None:
            return "", ""
        
        # DeepSeek模型使用的格式：thinking内容直接以</think>结束，没有开始标签
        if '</think>' in content:
            # 找到</think>标签的位置
            end_pos = content.find('</think>')
            thinking_content = content[:end_pos].strip()
            final_content = content[end_pos + 8:].strip()  # 8是'</think>'的长度
            return thinking_content, final_content
        
        # 如果没有thinking标签，返回原内容作为最终回答
        return "", content
    
    async def create_chat_completion(
        self, 
        request: ChatCompletionRequest
    ) -> ChatCompletionResponse:
        """创建聊天完成（非流式）"""
        try:
            # 转换消息格式
            hf_messages = self.convert_messages_to_hf_format(request.messages)
            
            # 调用Hugging Face API
            response = self.client.chat.completions.create(
                model=request.model,
                messages=hf_messages,
                temperature=request.temperature,
                top_p=request.top_p,
                max_tokens=request.max_tokens,
                stop=request.stop,
                stream=False
            )
            
            # 转换响应格式
            return self._convert_hf_response_to_openai(response, request.model, request)
            
        except Exception as e:
            logger.error(f"Error in create_chat_completion: {str(e)}")
            raise
    
    async def create_chat_completion_stream(
        self, 
        request: ChatCompletionRequest
    ) -> AsyncGenerator[str, None]:
        """创建流式聊天完成"""
        try:
            # 转换消息格式
            hf_messages = self.convert_messages_to_hf_format(request.messages)
            
            # 调用Hugging Face API（流式）
            stream = self.client.chat.completions.create(
                model=request.model,
                messages=hf_messages,
                temperature=request.temperature,
                top_p=request.top_p,
                max_tokens=request.max_tokens,
                stop=request.stop,
                stream=True
            )
            
            response_id = self.generate_response_id()
            created = self.get_current_timestamp()
            
            # 用于累积thinking内容的变量
            accumulated_content = ""
            in_thinking = False
            thinking_buffer = ""
            
            # 发送流式响应
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    accumulated_content += content
                    
                    # DeepSeek模型的thinking处理：没有开始标签，直接收集到</think>
                    # 如果还没有遇到</think>，默认认为是thinking内容
                    if not in_thinking and '</think>' not in accumulated_content:
                        # 开始时默认为thinking模式
                        in_thinking = True
                        thinking_buffer += content
                        continue
                    
                    if '</think>' in content and in_thinking:
                        # thinking结束
                        think_end_parts = content.split('</think>', 1)
                        thinking_buffer += think_end_parts[0]
                        
                        # 发送thinking内容
                        if thinking_buffer.strip():
                            stream_response = ChatCompletionStreamResponse(
                                id=response_id,
                                created=created,
                                model=request.model,
                                choices=[
                                    StreamChoice(
                                        index=0,
                                        delta=Delta(reasoning=thinking_buffer.strip()),
                                        finish_reason=None
                                    )
                                ]
                            )
                            yield f"data: {stream_response.model_dump_json()}\n\n"
                        
                        # 重置thinking状态
                        in_thinking = False
                        thinking_buffer = ""
                        
                        # 发送thinking之后的内容
                        after_think = think_end_parts[1] if len(think_end_parts) > 1 else ""
                        if after_think:
                            stream_response = ChatCompletionStreamResponse(
                                id=response_id,
                                created=created,
                                model=request.model,
                                choices=[
                                    StreamChoice(
                                        index=0,
                                        delta=Delta(content=after_think),
                                        finish_reason=chunk.choices[0].finish_reason
                                    )
                                ]
                            )
                            yield f"data: {stream_response.model_dump_json()}\n\n"
                        continue
                    
                    if in_thinking:
                        # 在thinking模式中，累积thinking内容
                        thinking_buffer += content
                        continue
                    
                    # 正常内容，直接发送
                    stream_response = ChatCompletionStreamResponse(
                        id=response_id,
                        created=created,
                        model=request.model,
                        choices=[
                            StreamChoice(
                                index=0,
                                delta=Delta(content=content),
                                finish_reason=chunk.choices[0].finish_reason
                            )
                        ]
                    )
                    yield f"data: {stream_response.model_dump_json()}\n\n"
                
                # 检查是否结束
                if chunk.choices and chunk.choices[0].finish_reason:
                    # 发送结束标记
                    final_response = ChatCompletionStreamResponse(
                        id=response_id,
                        created=created,
                        model=request.model,
                        choices=[
                            StreamChoice(
                                index=0,
                                delta=Delta(),
                                finish_reason=chunk.choices[0].finish_reason
                            )
                        ]
                    )
                    yield f"data: {final_response.model_dump_json()}\n\n"
                    break
            
            yield "data: [DONE]\n\n"
            
        except Exception as e:
            logger.error(f"Error in create_chat_completion_stream: {str(e)}")
            # 发送错误信息
            error_response = {
                "error": {
                    "message": str(e),
                    "type": "internal_error",
                    "code": "internal_error"
                }
            }
            yield f"data: {json.dumps(error_response)}\n\n"
    
    def _convert_hf_response_to_openai(self, hf_response, model: str, request: ChatCompletionRequest) -> ChatCompletionResponse:
        """将Hugging Face响应转换为OpenAI格式"""
        choice = hf_response.choices[0]
        
        # 解析thinking内容
        raw_content = choice.message.content or ""
        thinking_content, final_content = self.parse_thinking_content(raw_content)
        
        # 估算token使用量
        prompt_tokens = sum(self.estimate_tokens(msg.content) for msg in request.messages)
        completion_tokens = self.estimate_tokens(raw_content)
        
        return ChatCompletionResponse(
            id=self.generate_response_id(),
            created=self.get_current_timestamp(),
            model=model,
            choices=[
                Choice(
                    index=choice.index,
                    message=Message(
                        role=choice.message.role,
                        content=final_content,
                        reasoning=thinking_content if thinking_content else None
                    ),
                    finish_reason=choice.finish_reason
                )
            ],
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens
            )
        )
    
    async def get_models(self) -> ModelListResponse:
        """获取可用模型列表"""
        try:
            # 调用Hugging Face API获取模型列表
            models_response = self.client.models.list()
            
            models = []
            for model in models_response.data:
                models.append(Model(
                    id=model.id,
                    created=getattr(model, 'created', self.get_current_timestamp()),
                    owned_by=getattr(model, 'owned_by', 'huggingface')
                ))
            
            return ModelListResponse(data=models)
            
        except Exception as e:
            logger.error(f"Error in get_models: {str(e)}")
            # 返回默认模型列表
            return ModelListResponse(
                data=[
                    Model(
                        id=config.default_model,
                        created=self.get_current_timestamp(),
                        owned_by="huggingface"
                    )
                ]
            )