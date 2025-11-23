# GPT-4o 替换为阿里千问 (Qwen-VL) 说明

## ✅ 已完成的修改

### 1. 创建适配器文件
- **文件**: `utils/qwen_adapter.py`
- **功能**: 提供兼容OpenAI API格式的Qwen-VL适配器

### 2. 修改配置文件
- **文件**: `utils/config.py`
- **修改内容**:
  - 使用Qwen适配器替代OpenAI客户端
  - 默认API Key: `sk-79767f4057244ae5b52f4b5d0af2f01d`
  - 默认模型: `qwen-vl-max`

### 3. 修改运行文件
- **文件**: `run.py`
- **修改内容**:
  - 将模型从 `"gpt-4o"` 改为使用 `QWEN_MODEL`
  - 更新注释

### 4. 修改工具文件
- **文件**: `utils/utils.py`
- **修改内容**:
  - 将模型从 `"gpt-4o"` 改为使用 `QWEN_MODEL`

## 📦 需要安装的依赖

### 安装 dashscope SDK

```bash
pip install dashscope
```

或者在 `requirements.txt` 中添加：

```
dashscope
```

## 🔧 配置说明

### 默认配置（已硬编码）

- **API Key**: `sk-79767f4057244ae5b52f4b5d0af2f01d`
- **模型**: `qwen-vl-max`

### 通过环境变量覆盖（可选）

```bash
# 设置自定义API Key
export DASHSCOPE_API_KEY="sk-your-custom-key"

# 设置自定义模型
export QWEN_MODEL="qwen2-vl-72b-instruct"  # 或其他Qwen模型
```

## 🧪 测试

修改完成后，运行项目应该会自动使用Qwen-VL模型：

```bash
python run.py
```

或者运行评估：

```bash
python evaluate.py
```

## 📝 主要变化

1. **API调用方式**: 从OpenAI API改为阿里云DashScope API
2. **模型名称**: 从`gpt-4o`改为`qwen-vl-max`
3. **API格式**: 保持兼容，无需修改调用代码逻辑

## ⚠️ 注意事项

1. **API格式兼容**: Qwen适配器已处理格式转换，代码调用方式保持不变
2. **错误处理**: 如果API调用失败，会抛出异常，包含详细错误信息
3. **模型选择**: 可以通过环境变量`QWEN_MODEL`切换不同Qwen模型

## 🔗 相关资源

- [阿里云DashScope文档](https://help.aliyun.com/zh/model-studio/)
- [Qwen-VL模型文档](https://help.aliyun.com/zh/model-studio/developer-reference/qwen-vl-plus-api)

## 📞 支持

如果遇到问题，请检查：
1. `dashscope` 是否正确安装
2. API Key是否有效
3. 网络连接是否正常（需要访问阿里云API）

