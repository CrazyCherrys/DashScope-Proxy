import os
import json
import logging
import hashlib
from typing import List, Optional, Dict, Any, AsyncGenerator
from fastapi import FastAPI, HTTPException, Header, Request, UploadFile, File, Form
from fastapi.responses import StreamingResponse, JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import dashscope
from dashscope import Generation
import time
import uuid
import httpx

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
    max_tokens: Optional[int] = 2000
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

# NewAPI 视频生成请求格式
class VideoGenerationRequest(BaseModel):
    model: str
    prompt: Optional[str] = None
    image: Optional[str] = None  # 参考图/首帧
    duration: Optional[float] = None
    width: Optional[int] = None
    height: Optional[int] = None
    fps: Optional[int] = None
    seed: Optional[int] = None
    n: Optional[int] = 1
    response_format: Optional[str] = None
    user: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None  # 透传附加控制参数

# OpenAI 兼容视频响应
def build_openai_video_response(
    task_id: str,
    status: str,
    model: Optional[str] = None,
    created_at: Optional[int] = None,
    progress: Optional[int] = None,
    url: Optional[str] = None,
    format: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    error: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    resp = {
        "id": task_id,
        "object": "video",
        "model": model or "",
        "status": status,
        "progress": progress if progress is not None else (100 if status == "completed" else 0),
        "created_at": created_at if created_at is not None else int(time.time()),
    }
    if url:
        resp["url"] = url
    if format:
        resp["format"] = format
    if metadata:
        resp["metadata"] = metadata
    if error:
        resp["error"] = error
    return resp

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

def map_dashscope_task_status(status: Optional[str]) -> str:
    """将DashScope任务状态转换为NewAPI状态"""
    status = (status or "").upper()
    if status in ("PENDING", "QUEUED"):
        return "queued"
    if status in ("RUNNING", "DOING", "PRE-PROCESSING", "POST-PROCESSING", "PROCESSING"):
        return "in_progress"
    if status in ("SUCCEEDED", "SUCCESS", "COMPLETED", "FINISHED"):
        return "completed"
    if status in ("FAILED", "FAILURE", "CANCELED", "CANCELLED"):
        return "failed"
    return "queued"

def build_dashscope_video_payload(request: VideoGenerationRequest) -> Dict[str, Any]:
    """构建DashScope视频生成请求体"""
    input_block: Dict[str, Any] = {}
    params: Dict[str, Any] = {}

    if request.prompt:
        input_block["prompt"] = request.prompt
    if request.image:
        # DashScope I2V 使用 img_url
        input_block["img_url"] = request.image

    if request.duration is not None:
        params["duration"] = request.duration
    if request.width and request.height:
        params["size"] = f"{request.width}*{request.height}"
    if request.fps is not None:
        params["frame_rate"] = request.fps
    if request.seed is not None:
        params["seed"] = request.seed

    # 透传 metadata 中的已知控制字段
    if request.metadata:
        known_input_keys = {"negative_prompt", "prompt_extend", "reference_img", "reference_video_urls", "audio_url", "first_frame_url", "last_frame_url", "video_url"}
        for key, value in request.metadata.items():
            if value is None:
                continue
            if key in known_input_keys:
                input_block[key] = value
            else:
                params[key] = value

    payload: Dict[str, Any] = {
        "model": request.model,
        "input": input_block
    }
    if params:
        payload["parameters"] = params
    return payload

def parse_video_metadata(output: Dict[str, Any]) -> Dict[str, Any]:
    """抽取视频元信息，兼容不同返回字段"""
    meta: Dict[str, Any] = {}
    duration = output.get("duration") or output.get("video_duration")
    fps = output.get("frame_rate") or output.get("fps")
    size = output.get("size")
    resolution = output.get("resolution")

    if duration is not None:
        meta["duration"] = duration
    if fps is not None:
        meta["fps"] = fps

    width = height = None
    if isinstance(size, str) and "*" in size:
        try:
            width, height = [int(x) for x in size.split("*", 1)]
        except Exception:
            width = height = None
    if not width and isinstance(resolution, str) and "x" in resolution.lower():
        try:
            width, height = [int(x) for x in resolution.lower().split("x", 1)]
        except Exception:
            width = height = None
    if width is not None:
        meta["width"] = width
    if height is not None:
        meta["height"] = height

    seed = output.get("seed")
    if seed is not None:
        meta["seed"] = seed
    return meta

def extract_video_url(output: Dict[str, Any]) -> Optional[str]:
    """从DashScope返回中提取视频地址"""
    if "video_url" in output and output["video_url"]:
        return output["video_url"]
    if "url" in output and output["url"]:
        return output["url"]
    if "results" in output and isinstance(output["results"], list) and output["results"]:
        first = output["results"][0]
        if isinstance(first, dict):
            return first.get("url") or first.get("video_url")
    return None

def parse_size_to_wh(size: Optional[str]) -> (Optional[int], Optional[int]):
    if not size:
        return None, None
    sep = "x" if "x" in size.lower() else "*"
    try:
        parts = size.lower().split(sep)
        if len(parts) == 2:
            return int(parts[0]), int(parts[1])
    except Exception:
        return None, None
    return None, None

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
            },
            {
                "id": "qwen-plus",
                "object": "model", 
                "created": int(time.time()),
                "owned_by": "alibaba",
                "description": "阿里百炼 qwen-plus 文本生成模型"
            },
            {
                "id": "qwen-max",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "alibaba", 
                "description": "阿里百炼 qwen-max 文本生成模型"
            },
            {
                "id": "qwen-turbo",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "alibaba",
                "description": "阿里百炼 qwen-turbo 文本生成模型"
            }
        ]
    }

def truncate_conversation_history(messages: List[ChatMessage], max_messages: int = 20) -> List[ChatMessage]:
    """
    截断对话历史，保留最近的消息
    对于farui-plus模型，保持合理的对话长度以避免上下文溢出
    """
    if len(messages) <= max_messages:
        return messages
    
    # 总是保留system消息（如果存在）
    system_messages = [msg for msg in messages if msg.role == "system"]
    non_system_messages = [msg for msg in messages if msg.role != "system"]
    
    # 保留最近的对话
    remaining_slots = max_messages - len(system_messages)
    if len(non_system_messages) > remaining_slots:
        # 从最近的消息开始取，确保取偶数个以保持user-assistant配对
        recent_messages = non_system_messages[-remaining_slots:]
        
        # 如果第一条消息是assistant，删除它以保持对话流程
        if recent_messages and recent_messages[0].role == "assistant":
            recent_messages = recent_messages[1:]
    else:
        recent_messages = non_system_messages
    
    final_messages = system_messages + recent_messages
    return final_messages

def clean_empty_responses(messages: List[ChatMessage]) -> List[ChatMessage]:
    """
    清理空的assistant回复，避免恶性循环
    同时清理连续的重复user消息
    """
    cleaned_messages = []
    last_user_content = None
    
    for i, msg in enumerate(messages):
        # 跳过空的assistant回复
        if msg.role == "assistant" and (not msg.content or msg.content.strip() == ""):
            continue
            
        # 跳过连续的相同user消息（保留最后一个）
        if msg.role == "user":
            if msg.content == last_user_content:
                # 如果这不是最后一条消息，且下一条不是有效的assistant回复，则跳过
                if i < len(messages) - 1:
                    next_msg = messages[i + 1]
                    if next_msg.role == "assistant" and next_msg.content and next_msg.content.strip():
                        # 下一条是有效回复，保留这条user消息
                        pass
                    else:
                        # 下一条不是有效回复，跳过这条重复的user消息
                        continue
            last_user_content = msg.content
            
        cleaned_messages.append(msg)
    
    return cleaned_messages

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
        
        # 清理和截断对话历史
        cleaned_messages = clean_empty_responses(request.messages)
        truncated_messages = truncate_conversation_history(cleaned_messages, max_messages=20)
        
        # 记录对话历史处理信息
        if len(request.messages) != len(truncated_messages):
            logger.info(f"[{request_id}] Conversation history truncated: {len(request.messages)} -> {len(truncated_messages)} messages")
        
        # 转换消息格式
        dashscope_messages = convert_openai_to_dashscope_messages(truncated_messages)
        
        # 准备DashScope参数
        dashscope_params = {
            "model": request.model,
            "messages": dashscope_messages,
            "result_format": "message"
        }
        
        # 添加可选参数
        if request.temperature is not None:
            dashscope_params["temperature"] = request.temperature
        
        # 确保总是设置max_tokens，并验证范围
        if request.max_tokens is not None:
            # 为farui-plus模型验证max_tokens范围
            if request.model == "farui-plus":
                dashscope_params["max_tokens"] = min(max(request.max_tokens, 1), 2000)
            else:
                dashscope_params["max_tokens"] = min(max(request.max_tokens, 1), 4000)
        else:
            # 为不同模型设置合适的默认值
            if request.model == "farui-plus":
                dashscope_params["max_tokens"] = 2000  # farui-plus最大支持2000
            else:
                dashscope_params["max_tokens"] = 4000
                
        if request.top_p is not None:
            dashscope_params["top_p"] = request.top_p
        if request.stop is not None:
            dashscope_params["stop"] = request.stop
            
        # 添加详细的参数日志
        logger.info(f"[{request_id}] DashScope params: {json.dumps(dashscope_params, ensure_ascii=False, indent=2)}")
        logger.info(f"[{request_id}] Calling DashScope API - IP: {client_ip}, API Key Hash: {api_key_hash}, Model: {request.model}, Messages Count: {len(truncated_messages)} (original: {len(request.messages)}), Stream: {request.stream}")
        
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
            
            # 添加响应内容的详细日志
            content = response.output.choices[0].message.content
            logger.info(f"[{request_id}] Response content length: {len(content)} characters")
            logger.info(f"[{request_id}] Response content preview: {content[:100]}...")
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
        dashscope_params["incremental_output"] = True  # 启用增量输出
        
        logger.info(f"[{request_id}] Starting stream response")
        
        responses = Generation.call(**dashscope_params)
        
        chunk_count = 0
        accumulated_content = ""
        
        for response in responses:
            chunk_count += 1
            logger.info(f"[{request_id}] Processing chunk {chunk_count}")
            
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
            
            # 处理DashScope的流式响应
            if hasattr(response.output, 'choices') and response.output.choices:
                choice = response.output.choices[0]
                
                # 获取当前内容
                current_content = ""
                finish_reason = None
                
                # 尝试多种方式获取内容
                if hasattr(choice, 'message') and choice.message and hasattr(choice.message, 'content'):
                    current_content = choice.message.content or ""
                    finish_reason = getattr(choice, 'finish_reason', None)
                    
                    # 计算增量内容（DashScope可能返回累积内容）
                    if current_content.startswith(accumulated_content):
                        # 如果当前内容包含之前的内容，提取增量
                        incremental_content = current_content[len(accumulated_content):]
                    else:
                        # 否则使用全部内容作为增量
                        incremental_content = current_content
                        
                    accumulated_content = current_content
                    
                elif hasattr(choice, 'delta') and choice.delta:
                    # 直接处理增量内容
                    if isinstance(choice.delta, dict):
                        incremental_content = choice.delta.get('content', '')
                    else:
                        incremental_content = getattr(choice.delta, 'content', '') if hasattr(choice.delta, 'content') else ""
                    
                    accumulated_content += incremental_content
                    finish_reason = getattr(choice, 'finish_reason', None)
                else:
                    # 如果没有找到内容，跳过这个chunk
                    logger.warning(f"[{request_id}] Chunk {chunk_count} has no content")
                    continue
                
                logger.info(f"[{request_id}] Chunk {chunk_count} incremental content length: {len(incremental_content)}")
                if incremental_content:
                    logger.info(f"[{request_id}] Chunk {chunk_count} content preview: {incremental_content[:50]}...")
                
                # 只有当有增量内容或者是结束chunk时才发送
                if incremental_content or (finish_reason and finish_reason != "null"):
                    # 构建流式响应chunk
                    delta = {}
                    if incremental_content:
                        delta["content"] = incremental_content
                    
                    stream_chunk = ChatCompletionStreamResponse(
                        id=completion_id,
                        created=created_time,
                        model=model,
                        choices=[ChatCompletionStreamChoice(
                            index=0,
                            delta=delta,
                            finish_reason=finish_reason if finish_reason != "null" else None
                        )]
                    )
                    
                    chunk_json = stream_chunk.model_dump_json()
                    logger.info(f"[{request_id}] Sending chunk {chunk_count}: {chunk_json[:150]}...")
                    yield f"data: {chunk_json}\n\n"
                    
                    if finish_reason and finish_reason != "null":
                        logger.info(f"[{request_id}] Stream finished with reason: {finish_reason}")
                        break
            else:
                logger.warning(f"[{request_id}] Chunk {chunk_count} has no valid choices")
        
        logger.info(f"[{request_id}] Stream completed successfully - Chunks: {chunk_count}")
        
        # 发送结束标记
        yield "data: [DONE]\n\n"
        
    except Exception as e:
        logger.error(f"[{request_id}] Stream error: {str(e)}", exc_info=True)
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

@app.post("/v1/video/generations")
async def create_video_generation(
    request: VideoGenerationRequest,
    authorization: Optional[str] = Header(None)
):
    """兼容NewAPI的视频生成创建接口，转发到DashScope异步视频生成"""
    request_id = f"vid-{uuid.uuid4().hex[:8]}"
    api_key = get_api_key_from_header(authorization)
    api_key_hash = hash_api_key(api_key)

    dashscope_api = f"{DASHSCOPE_BASE_URL}/api/v1/services/aigc/video-generation/video-synthesis"
    payload = build_dashscope_video_payload(request)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "X-DashScope-Async": "enable"
    }

    logger.info(f"[{request_id}] Create video task - Model: {request.model}, API Key Hash: {api_key_hash}")
    logger.info(f"[{request_id}] DashScope video payload: {json.dumps(payload, ensure_ascii=False)}")

    async with httpx.AsyncClient(timeout=180) as client:
        resp = await client.post(dashscope_api, headers=headers, json=payload)

    if resp.status_code != 200:
        try:
            detail = resp.json()
            message = detail.get("message") or detail.get("error", {}).get("message") or resp.text
            code = detail.get("code") or detail.get("error", {}).get("code")
        except Exception:
            detail = None
            message = resp.text
            code = "dashscope_error"
        logger.error(f"[{request_id}] DashScope video API error {resp.status_code}: {message}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": {
                    "message": f"DashScope video API error: {message}",
                    "type": "dashscope_error",
                    "code": code or resp.status_code
                }
            }
        )

    resp_json = resp.json()
    output = resp_json.get("output", {})
    task_id = output.get("task_id") or resp_json.get("task_id")
    task_status = output.get("task_status") or output.get("status") or resp_json.get("task_status") or resp_json.get("status")

    if not task_id:
        logger.error(f"[{request_id}] Missing task_id in DashScope response: {resp_json}")
        raise HTTPException(
            status_code=500,
            detail={"error": {"message": "DashScope response missing task_id", "type": "dashscope_error", "code": "missing_task_id"}}
        )

    status_mapped = map_dashscope_task_status(task_status)
    return {"task_id": task_id, "status": status_mapped}

@app.get("/v1/video/generations/{task_id}")
async def get_video_generation(
    task_id: str,
    authorization: Optional[str] = Header(None)
):
    """兼容NewAPI的视频任务查询接口，轮询DashScope任务状态"""
    request_id = f"vidq-{uuid.uuid4().hex[:8]}"
    api_key = get_api_key_from_header(authorization)
    api_key_hash = hash_api_key(api_key)

    url = f"{DASHSCOPE_BASE_URL}/api/v1/tasks/{task_id}"
    headers = {"Authorization": f"Bearer {api_key}"}

    logger.info(f"[{request_id}] Query video task {task_id} - API Key Hash: {api_key_hash}")

    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.get(url, headers=headers)

    if resp.status_code != 200:
        try:
            detail = resp.json()
            message = detail.get("message") or detail.get("error", {}).get("message") or resp.text
            code = detail.get("code") or detail.get("error", {}).get("code")
        except Exception:
            detail = None
            message = resp.text
            code = "dashscope_error"
        logger.error(f"[{request_id}] DashScope task query error {resp.status_code}: {message}")
        raise HTTPException(
            status_code=resp.status_code if resp.status_code != 401 else 401,
            detail={
                "error": {
                    "message": f"DashScope task query error: {message}",
                    "type": "dashscope_error",
                    "code": code or resp.status_code
                }
            }
        )

    resp_json = resp.json()
    logger.info(f"[{request_id}] Task query raw response: {json.dumps(resp_json, ensure_ascii=False)[:1000]}")
    output = resp_json.get("output", {})
    task_status = output.get("task_status") or output.get("status") or resp_json.get("task_status") or resp_json.get("status")
    status_mapped = map_dashscope_task_status(task_status)
    video_url = extract_video_url(output)
    metadata = parse_video_metadata(output)

    # 若已拿到视频地址但状态仍未完成，主动提升为 completed，避免队列状态卡住
    if video_url and status_mapped in ("queued", "in_progress"):
        status_mapped = "completed"

    # 补充格式信息
    video_format = None
    if video_url and "." in video_url.split("/")[-1]:
        ext = video_url.split("/")[-1].split(".")[-1]
        if ext:
            video_format = ext

    error_block = None
    if status_mapped == "failed":
        message = output.get("message") or resp_json.get("message")
        error_block = {"code": output.get("code") or resp_json.get("code"), "message": message}

    response_body = {
        "task_id": task_id,
        "status": status_mapped,
        "url": video_url,
        "format": video_format,
        "metadata": metadata or None,
        "error": error_block
    }

    return response_body

# 兼容 NewAPI 默认路由 `/v1/videos` 的别名
@app.post("/v1/videos")
async def create_video_generation_alias(
    request: Request,
    authorization: Optional[str] = Header(None),
    prompt: Optional[str] = Form(None),
    model: Optional[str] = Form(None),
    seconds: Optional[str] = Form(None),
    size: Optional[str] = Form(None),
    input_reference: Optional[UploadFile] = File(None),
    metadata: Optional[str] = Form(None)
):
    """
    OpenAI 兼容视频创建端点，支持 multipart/form-data（OpenAI Sora 格式）和 JSON 两种方式。
    """
    content_type = request.headers.get("content-type", "").lower()

    # multipart/form-data 分支
    if "multipart/form-data" in content_type:
        meta_dict = None
        if metadata:
            try:
                meta_dict = json.loads(metadata)
            except Exception:
                raise HTTPException(status_code=400, detail={"error": {"message": "metadata 必须是 JSON 字符串"}})

        width, height = parse_size_to_wh(size)

        # 文件暂不支持直接上传到 DashScope，提示用户使用 URL
        if input_reference:
            raise HTTPException(
                status_code=400,
                detail={"error": {"message": "当前版本暂不支持直接上传文件，请提供可访问的图像/视频 URL（metadata.reference_img 或 metadata.reference_video_urls）"}}
            )

        req = VideoGenerationRequest(
            model=model or "wan2.6-t2v",
            prompt=prompt,
            duration=float(seconds) if seconds else None,
            width=width,
            height=height,
            metadata=meta_dict
        )
        base_resp = await create_video_generation(req, authorization)
        return build_openai_video_response(
            task_id=base_resp.get("task_id"),
            status=base_resp.get("status"),
            model=req.model,
            created_at=int(time.time())
        )

    # JSON 分支（兼容旧调用）
    body = await request.json()
    try:
        req = VideoGenerationRequest(**body)
    except Exception as e:
        raise HTTPException(status_code=400, detail={"error": {"message": f"请求体解析失败: {str(e)}"}})
    base_resp = await create_video_generation(req, authorization)
    return build_openai_video_response(
        task_id=base_resp.get("task_id"),
        status=base_resp.get("status"),
        model=req.model,
        created_at=int(time.time())
    )

@app.get("/v1/videos/{task_id}")
async def get_video_generation_alias(
    task_id: str,
    authorization: Optional[str] = Header(None)
):
    base_resp = await get_video_generation(task_id, authorization)
    return build_openai_video_response(
        task_id=task_id,
        status=base_resp.get("status"),
        model=base_resp.get("model"),
        created_at=int(time.time()),
        url=base_resp.get("url"),
        format=base_resp.get("format"),
        metadata=base_resp.get("metadata"),
        error=base_resp.get("error")
    )

@app.get("/v1/videos/{task_id}/content")
async def get_video_content(
    task_id: str,
    authorization: Optional[str] = Header(None)
):
    """
    OpenAI 兼容的视频内容获取接口：返回视频二进制流。
    """
    # 复用查询逻辑，确保同一 key 与域名
    api_key = get_api_key_from_header(authorization)
    url = f"{DASHSCOPE_BASE_URL}/api/v1/tasks/{task_id}"
    headers = {"Authorization": f"Bearer {api_key}"}

    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.get(url, headers=headers)
    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail={"error": {"message": resp.text}})

    data = resp.json()
    output = data.get("output", {})
    video_url = extract_video_url(output)
    status = map_dashscope_task_status(output.get("task_status") or output.get("status") or data.get("task_status"))
    if status != "completed" or not video_url:
        raise HTTPException(status_code=404, detail={"error": {"message": "Video not ready"}})

    # 代理视频流
    async with httpx.AsyncClient(timeout=None) as client:
        upstream = await client.get(video_url, follow_redirects=True)
        if upstream.status_code != 200:
            raise HTTPException(status_code=upstream.status_code, detail={"error": {"message": "Failed to fetch video content"}})
        content_type = upstream.headers.get("content-type", "video/mp4")
        return StreamingResponse(upstream.aiter_bytes(), media_type=content_type)

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
