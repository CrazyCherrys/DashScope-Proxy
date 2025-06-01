#!/usr/bin/env python3
"""
DashScope to OpenAI API Gateway 测试客户端

这是一个公共服务的测试示例。
用户需要提供自己的DashScope API密钥来使用服务。
"""

import openai
import json
import os
import sys

# 设置网关服务器地址
GATEWAY_BASE_URL = "http://localhost:8000/v1"

def get_api_key():
    """获取用户的DashScope API密钥"""
    # 首先尝试从环境变量获取
    api_key = os.getenv('DASHSCOPE_API_KEY')
    
    if not api_key:
        # 如果环境变量没有，提示用户输入
        print("=" * 60)
        print("🔑 DashScope API 密钥配置")
        print("=" * 60)
        print("请提供您的DashScope API密钥来使用这个公共服务。")
        print("您可以：")
        print("1. 设置环境变量: export DASHSCOPE_API_KEY=your_api_key")
        print("2. 或者在下面直接输入（仅用于测试）")
        print()
        
        api_key = input("请输入您的DashScope API密钥: ").strip()
        
        if not api_key:
            print("❌ 未提供API密钥，退出测试。")
            sys.exit(1)
    
    return api_key

def test_service_info():
    """测试服务信息"""
    print("=== 📊 服务信息 ===")
    
    import requests
    try:
        response = requests.get("http://localhost:8000/")
        if response.status_code == 200:
            info = response.json()
            print(f"✅ 服务状态: {info['message']}")
            print(f"📝 描述: {info['description']}")
            print(f"🔖 版本: {info['version']}")
            print(f"📍 API端点: {info['usage']['endpoint']}")
            print()
        else:
            print(f"❌ 无法获取服务信息，状态码: {response.status_code}")
    except Exception as e:
        print(f"❌ 连接失败: {e}")
        print("请确保服务正在运行: python main.py")
        return False
    return True

def test_models_endpoint(api_key):
    """测试模型列表端点"""
    print("=== 📋 可用模型列表 ===")
    
    client = openai.OpenAI(
        api_key=api_key,
        base_url=GATEWAY_BASE_URL
    )
    
    try:
        models = client.models.list()
        print("可用模型:")
        for model in models.data:
            print(f"  🤖 {model.id}")
            print(f"     所有者: {model.owned_by}")
            if hasattr(model, 'description'):
                print(f"     描述: {model.description}")
        print()
        return True
        
    except Exception as e:
        print(f"❌ 获取模型列表失败: {e}")
        return False

def test_non_stream_chat(api_key):
    """测试非流式聊天"""
    print("=== 💬 非流式聊天测试 ===")
    
    client = openai.OpenAI(
        api_key=api_key,
        base_url=GATEWAY_BASE_URL
    )
    
    try:
        print("发送消息: 请简单介绍一下人工智能...")
        
        response = client.chat.completions.create(
            model="farui-plus",
            messages=[
                {"role": "system", "content": "你是一个有用的AI助手。请用简洁的语言回答问题。"},
                {"role": "user", "content": "请简单介绍一下人工智能，控制在100字以内。"}
            ],
            temperature=0.7,
            max_tokens=200
        )
        
        print(f"✅ 响应ID: {response.id}")
        print(f"🤖 模型: {response.model}")
        print(f"💭 回复: {response.choices[0].message.content}")
        print(f"📊 Token使用: {response.usage}")
        print()
        return True
        
    except Exception as e:
        print(f"❌ 非流式聊天失败: {e}")
        return False

def test_stream_chat(api_key):
    """测试流式聊天"""
    print("=== 🌊 流式聊天测试 ===")
    
    client = openai.OpenAI(
        api_key=api_key,
        base_url=GATEWAY_BASE_URL
    )
    
    try:
        print("发送消息: 写一首关于科技的短诗...")
        print("流式响应:")
        print("-" * 40)
        
        stream = client.chat.completions.create(
            model="farui-plus",
            messages=[
                {"role": "system", "content": "你是一个有创意的诗人。"},
                {"role": "user", "content": "请写一首关于科技与未来的短诗，4行即可。"}
            ],
            temperature=0.8,
            stream=True
        )
        
        full_response = ""
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                content = chunk.choices[0].delta.content
                print(content, end='', flush=True)
                full_response += content
        
        print()
        print("-" * 40)
        print("✅ 流式响应完成")
        print()
        return True
        
    except Exception as e:
        print(f"❌ 流式聊天失败: {e}")
        return False

def show_usage_examples():
    """显示使用示例"""
    print("=== 📚 使用示例 ===")
    
    print("1. Python (OpenAI SDK):")
    print("""
import openai

client = openai.OpenAI(
    api_key="您的DashScope-API-Key",
    base_url="http://您的服务器地址:8000/v1"
)

response = client.chat.completions.create(
    model="farui-plus",
    messages=[{"role": "user", "content": "你好！"}]
)
""")
    
    print("2. curl 命令:")
    print(f"""
curl -X POST "{GATEWAY_BASE_URL}/chat/completions" \\
  -H "Content-Type: application/json" \\
  -H "Authorization: Bearer 您的DashScope-API-Key" \\
  -d '{{
    "model": "farui-plus",
    "messages": [
      {{"role": "user", "content": "你好！"}}
    ]
  }}'
""")

def main():
    """主函数"""
    print("🚀 DashScope to OpenAI API Gateway 测试客户端")
    print("=" * 60)
    print("这是一个公共服务，您需要提供自己的DashScope API密钥。")
    print("=" * 60)
    print()
    
    # 检查服务状态
    if not test_service_info():
        print("❌ 服务不可用，请检查服务是否启动。")
        return
    
    # 获取API密钥
    try:
        api_key = get_api_key()
        print(f"✅ API密钥已配置 (前8位: {api_key[:8]}...)")
        print()
    except KeyboardInterrupt:
        print("\n用户取消操作。")
        return
    
    # 运行测试
    success_count = 0
    total_tests = 3
    
    if test_models_endpoint(api_key):
        success_count += 1
    
    if test_non_stream_chat(api_key):
        success_count += 1
    
    if test_stream_chat(api_key):
        success_count += 1
    
    # 显示结果
    print("=" * 60)
    print(f"📊 测试结果: {success_count}/{total_tests} 项测试通过")
    
    if success_count == total_tests:
        print("🎉 所有测试通过！服务运行正常。")
    else:
        print("⚠️  部分测试失败，请检查配置和网络连接。")
    
    print("=" * 60)
    
    # 显示使用示例
    show_usage_examples()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 测试被用户中断。") 