"""
阿里千问API适配器，兼容OpenAI API格式
"""
import dashscope
from dashscope import MultiModalConversation
import base64
import os


class QwenAdapter:
    """阿里千问API适配器类"""
    
    def __init__(self, api_key=None):
        """
        初始化适配器
        
        Args:
            api_key: DashScope API密钥，如果不提供则从环境变量读取
        """
        if api_key:
            dashscope.api_key = api_key
        elif os.environ.get("DASHSCOPE_API_KEY"):
            dashscope.api_key = os.environ["DASHSCOPE_API_KEY"]
        else:
            raise ValueError("DASHSCOPE_API_KEY not found in environment variables")
    
    class ChatCompletion:
        """兼容OpenAI ChatCompletion接口"""
        
        @staticmethod
        def create(model="qwen-vl-max", messages=None, **kwargs):
            """
            兼容OpenAI格式的调用接口
            
            Args:
                model: 模型名称 (qwen-vl-max, qwen2-vl-72b-instruct, qwen-vl-plus)
                messages: 消息列表，格式与OpenAI兼容
                **kwargs: 其他参数（temperature, max_tokens等）
            
            Returns:
                兼容OpenAI格式的响应对象
            """
            # 转换消息格式为Qwen-VL格式
            qwen_messages = []
            for msg in messages:
                if msg["role"] == "system":
                    # Qwen-VL支持system消息
                    qwen_messages.append({
                        "role": "system",
                        "content": msg["content"]
                    })
                elif msg["role"] == "user":
                    # 处理多模态内容
                    content = []
                    if isinstance(msg["content"], str):
                        # 如果content是字符串
                        content.append({"text": msg["content"]})
                    elif isinstance(msg["content"], list):
                        # 如果content是列表（包含文本和图片）
                        for item in msg["content"]:
                            if item["type"] == "text":
                                content.append({
                                    "text": item["text"]
                                })
                            elif item["type"] == "image_url":
                                # 提取base64图片
                                image_url = item["image_url"]["url"]
                                if image_url.startswith("data:image"):
                                    # 提取base64部分
                                    base64_data = image_url.split(",", 1)[1]
                                    content.append({
                                        "image": base64_data
                                    })
                                elif image_url.startswith("http"):
                                    # 如果是URL，直接使用
                                    content.append({
                                        "image": image_url
                                    })
                                else:
                                    # 假设是base64字符串
                                    content.append({
                                        "image": image_url
                                    })
                    
                    qwen_messages.append({
                        "role": "user",
                        "content": content
                    })
            
            # 调用Qwen-VL API
            try:
                response = MultiModalConversation.call(
                    model=model,
                    messages=qwen_messages,
                    temperature=kwargs.get("temperature", 0),
                    max_tokens=kwargs.get("max_tokens", 2000),
                    top_p=kwargs.get("top_p", 1.0),
                )
                
                # 转换为OpenAI格式的响应
                if response.status_code == 200:
                    output_text = response.output.choices[0].message.content
                    
                    # 创建兼容OpenAI格式的响应对象
                    class Choice:
                        class Message:
                            def __init__(self, content):
                                self.content = content
                        
                        def __init__(self, content):
                            self.message = self.Message(content)
                    
                    class Response:
                        def __init__(self, choices):
                            self.choices = choices
                    
                    return Response([Choice(output_text)])
                else:
                    error_msg = f"Qwen API Error (status_code: {response.status_code}): {response.message}"
                    raise Exception(error_msg)
                    
            except Exception as e:
                raise Exception(f"Qwen API call failed: {str(e)}")

