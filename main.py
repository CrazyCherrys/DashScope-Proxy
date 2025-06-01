import os
import json
import logging
import hashlib
from typing import List, Optional, Dict, Any, AsyncGenerator
from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import dashscope
from dashscope import Generation
import time
import uuid

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# DashScope 配置
DASHSCOPE_BASE_URL = os.getenv('DASHSCOPE_BASE_URL', 'https://dashscope.aliyuncs.com')

app = FastAPI(
    title="DashScope to OpenAI API Gateway",
    description="将阿里百炼DashScope API转换为OpenAI API格式的中间件 - 公共服务",
    version="1.0.0"
)

# 添加CORS支持
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# OpenAI API 请求格式
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = None
    top_p: Optional[float] = 1.0
    stream: Optional[bool] = False
    stop: Optional[List[str]] = None

# OpenAI API 响应格式
class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Optional[str] = None

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: Dict[str, int]

# 流式响应格式
class ChatCompletionStreamChoice(BaseModel):
    index: int
    delta: Dict[str, Any]
    finish_reason: Optional[str] = None

class ChatCompletionStreamResponse(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[ChatCompletionStreamChoice]

def hash_api_key(api_key: str) -> str:
    """对API密钥进行哈希处理，用于日志记录（保护隐私）"""
    return hashlib.md5(api_key.encode()).hexdigest()[:8]

def convert_openai_to_dashscope_messages(messages: List[ChatMessage]) -> List[Dict[str, str]]:
    """将OpenAI格式的messages转换为DashScope格式"""
    dashscope_messages = []
    for message in messages:
        role_mapping = {
            "system": "system",
            "user": "user", 
            "assistant": "assistant"
        }
        dashscope_messages.append({
            "role": role_mapping.get(message.role, message.role),
            "content": message.content
        })
    return dashscope_messages

def get_api_key_from_header(authorization: Optional[str] = None) -> str:
    """从Header中提取API Key"""
    if not authorization:
        raise HTTPException(
            status_code=401, 
            detail={
                "error": {
                    "message": "Missing authorization header. Please provide your DashScope API key in the Authorization header as 'Bearer YOUR_API_KEY'",
                    "type": "authentication_error",
                    "code": "missing_authorization"
                }
            }
        )
    
    if authorization.startswith("Bearer "):
        api_key = authorization[7:]
        if not api_key or len(api_key) < 10:  # 基本的API key长度检查
            raise HTTPException(
                status_code=401, 
                detail={
                    "error": {
                        "message": "Invalid API key format. Please check your DashScope API key.",
                        "type": "authentication_error",
                        "code": "invalid_api_key"
                    }
                }
            )
        return api_key
    else:
        raise HTTPException(
            status_code=401, 
            detail={
                "error": {
                    "message": "Invalid authorization format. Use 'Bearer YOUR_API_KEY'",
                    "type": "authentication_error",
                    "code": "invalid_authorization_format"
                }
            }
        )

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """请求日志中间件"""
    start_time = time.time()
    
    # 获取客户端IP
    client_ip = request.client.host if request.client else "unknown"
    
    # 获取API key hash用于日志（不记录真实密钥）
    auth_header = request.headers.get("authorization", "")
    api_key_hash = "none"
    if auth_header.startswith("Bearer "):
        try:
            api_key = auth_header[7:]
            api_key_hash = hash_api_key(api_key)
        except:
            pass
    
    # 记录请求开始
    logger.info(f"Request started - IP: {client_ip}, Method: {request.method}, Path: {request.url.path}, API Key Hash: {api_key_hash}")
    
    response = await call_next(request)
    
    # 计算处理时间
    process_time = time.time() - start_time
    
    # 记录请求完成
    logger.info(f"Request completed - IP: {client_ip}, Status: {response.status_code}, Time: {process_time:.2f}s, API Key Hash: {api_key_hash}")
    
    return response

@app.get("/")
async def root():
    return {
        "message": "DashScope to OpenAI API Gateway is running",
        "version": "1.0.0",
        "description": "公共服务 - 将阿里百炼DashScope API转换为OpenAI API格式",
        "dashscope_endpoint": DASHSCOPE_BASE_URL,
        "usage": {
            "endpoint": "/v1/chat/completions",
            "method": "POST",
            "auth": "Bearer YOUR_DASHSCOPE_API_KEY",
            "model": "farui-plus"
        },
        "docs": "/docs"
    }

@app.get("/v1/models")
async def list_models(authorization: Optional[str] = Header(None)):
    """列出可用模型"""
    # 验证API Key（即使只是列出模型也需要验证）
    get_api_key_from_header(authorization)
    
    return {
        "object": "list",
        "data": [
            {
                "id": "farui-plus",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "alibaba",
                "description": "阿里百炼 farui-plus 文本生成模型"
            }
        ]
    }

@app.post("/v1/chat/completions")
async def create_chat_completion(
    request: ChatCompletionRequest,
    authorization: Optional[str] = Header(None),
    client_request: Request = None
):
    """创建聊天完成"""
    request_id = f"req-{uuid.uuid4().hex[:8]}"
    client_ip = client_request.client.host if client_request and client_request.client else "unknown"
    
    try:
        # 获取并验证API Key
        api_key = get_api_key_from_header(authorization)
        api_key_hash = hash_api_key(api_key)
        
        # 设置DashScope API Key
        dashscope.api_key = api_key
        
        # 设置自定义的base_url（如果指定了的话）
        if DASHSCOPE_BASE_URL != 'https://dashscope.aliyuncs.com':
            dashscope.base_http_api_url = DASHSCOPE_BASE_URL
        
        # 转换消息格式
        dashscope_messages = convert_openai_to_dashscope_messages(request.messages)
        
        # 准备DashScope参数
        dashscope_params = {
            "model": "farui-plus",
            "messages": dashscope_messages,
            "result_format": "message"
        }
        
        # 添加可选参数
        if request.temperature is not None:
            dashscope_params["temperature"] = request.temperature
        if request.max_tokens is not None:
            dashscope_params["max_tokens"] = request.max_tokens
        if request.top_p is not None:
            dashscope_params["top_p"] = request.top_p
        if request.stop is not None:
            dashscope_params["stop"] = request.stop
            
        logger.info(f"[{request_id}] Calling DashScope API - IP: {client_ip}, API Key Hash: {api_key_hash}, Model: {request.model}, Messages Count: {len(request.messages)}, Stream: {request.stream}")
        
        if request.stream:
            # 流式响应
            return StreamingResponse(
                stream_chat_completion(dashscope_params, request.model, request_id, api_key_hash),
                media_type="text/plain"
            )
        else:
            # 非流式响应
            response = Generation.call(**dashscope_params)
            
            if response.status_code != 200:
                logger.error(f"[{request_id}] DashScope API error - Code: {response.status_code}, Message: {response.message}")
                raise HTTPException(
                    status_code=500, 
                    detail={
                        "error": {
                            "message": f"DashScope API error: {response.message}",
                            "type": "dashscope_error",
                            "code": response.code if hasattr(response, 'code') else "unknown"
                        }
                    }
                )
            
            # 转换为OpenAI格式
            completion_id = f"chatcmpl-{uuid.uuid4().hex}"
            created_time = int(time.time())
            
            choice = ChatCompletionChoice(
                index=0,
                message=ChatMessage(
                    role="assistant",
                    content=response.output.choices[0].message.content
                ),
                finish_reason=response.output.choices[0].finish_reason
            )
            
            usage = {
                "prompt_tokens": response.usage.input_tokens if hasattr(response.usage, 'input_tokens') else 0,
                "completion_tokens": response.usage.output_tokens if hasattr(response.usage, 'output_tokens') else 0,
                "total_tokens": response.usage.total_tokens if hasattr(response.usage, 'total_tokens') else 0
            }
            
            logger.info(f"[{request_id}] Request completed successfully - Tokens: {usage['total_tokens']}")
            
            return ChatCompletionResponse(
                id=completion_id,
                created=created_time,
                model=request.model,
                choices=[choice],
                usage=usage
            )
            
    except HTTPException:
        # 重新抛出已经格式化的HTTP异常
        raise
    except Exception as e:
        logger.error(f"[{request_id}] Unexpected error: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail={
                "error": {
                    "message": f"Internal server error: {str(e)}",
                    "type": "internal_error",
                    "code": "server_error"
                }
            }
        )

async def stream_chat_completion(dashscope_params: Dict[str, Any], model: str, request_id: str, api_key_hash: str) -> AsyncGenerator[str, None]:
    """流式聊天完成生成器"""
    try:
        completion_id = f"chatcmpl-{uuid.uuid4().hex}"
        created_time = int(time.time())
        
        # 启用流式响应
        dashscope_params["stream"] = True
        
        logger.info(f"[{request_id}] Starting stream response")
        
        responses = Generation.call(**dashscope_params)
        
        chunk_count = 0
        for response in responses:
            chunk_count += 1
            
            if response.status_code != 200:
                logger.error(f"[{request_id}] Stream error - Code: {response.status_code}, Message: {response.message}")
                error_chunk = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created_time,
                    "model": model,
                    "choices": [{
                        "index": 0,
                        "delta": {},
                        "finish_reason": "error"
                    }],
                    "error": {"message": response.message}
                }
                yield f"data: {json.dumps(error_chunk)}\n\n"
                return
            
            # 转换为OpenAI流式格式
            if hasattr(response.output, 'choices') and response.output.choices:
                choice = response.output.choices[0]
                
                if hasattr(choice, 'message') and choice.message:
                    delta = {"content": choice.message.content}
                    finish_reason = choice.finish_reason if hasattr(choice, 'finish_reason') else None
                else:
                    delta = {}
                    finish_reason = None
                
                stream_chunk = ChatCompletionStreamResponse(
                    id=completion_id,
                    created=created_time,
                    model=model,
                    choices=[ChatCompletionStreamChoice(
                        index=0,
                        delta=delta,
                        finish_reason=finish_reason
                    )]
                )
                
                yield f"data: {stream_chunk.model_dump_json()}\n\n"
                
                if finish_reason:
                    break
        
        logger.info(f"[{request_id}] Stream completed successfully - Chunks: {chunk_count}")
        
        # 发送结束标记
        yield "data: [DONE]\n\n"
        
    except Exception as e:
        logger.error(f"[{request_id}] Stream error: {str(e)}")
        error_chunk = {
            "id": completion_id,
            "object": "chat.completion.chunk", 
            "created": created_time,
            "model": model,
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": "error"
            }],
            "error": {"message": str(e)}
        }
        yield f"data: {json.dumps(error_chunk)}\n\n"

@app.get("/health")
async def health_check():
    """健康检查端点"""
    return {
        "status": "healthy",
        "timestamp": int(time.time()),
        "service": "DashScope to OpenAI API Gateway"
    }

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """HTTP异常处理器"""
    client_ip = request.client.host if request.client else "unknown"
    logger.warning(f"HTTP Exception - IP: {client_ip}, Status: {exc.status_code}, Detail: {exc.detail}")
    
    # 如果detail已经是正确格式，直接返回
    if isinstance(exc.detail, dict) and "error" in exc.detail:
        return JSONResponse(
            status_code=exc.status_code,
            content=exc.detail
        )
    
    # 否则包装成标准格式
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "message": str(exc.detail),
                "type": "http_error",
                "code": str(exc.status_code)
            }
        }
    )

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """全局异常处理器"""
    client_ip = request.client.host if request.client else "unknown"
    logger.error(f"Global Exception - IP: {client_ip}, Error: {str(exc)}", exc_info=True)
    
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "message": "Internal server error occurred",
                "type": "internal_error",
                "code": "server_error"
            }
        }
    )

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting DashScope to OpenAI API Gateway...")
    uvicorn.run(app, host="0.0.0.0", port=8000) 