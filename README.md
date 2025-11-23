# API Key 配置说明

本项目支持 **OpenAI** 和 **智谱AI (ZhipuAI)** 两种API提供商，你可以根据需要选择使用。

## 快速开始（智谱AI）

1. 打开项目根目录下的 `config.py` 文件
2. 确认 `ZHIPUAI_API_KEY` 已设置（已为你配置好）
3. 确认 `API_PROVIDER = "zhipuai"`（默认已设置）
4. 直接运行脚本即可

## API提供商选择

在 `config.py` 中设置 `API_PROVIDER`：

- `"zhipuai"` - 使用智谱AI（默认）
- `"openai"` - 使用OpenAI

## 配置方式（按优先级从高到低）

### 方式一：使用配置文件（推荐）

1. 打开项目根目录下的 `config.py` 文件
2. 根据选择的API提供商，设置对应的API Key：

**智谱AI配置：**

```python
ZHIPUAI_API_KEY = "78443ce13bdd4f72809efda1abd95af4.s6TNCSWw3Q4Ci2wx"
API_PROVIDER = "zhipuai"
DEFAULT_MODEL_NAME = "glm-4-flash-250414"
```

**OpenAI配置：**

```python
OPENAI_API_KEY = "sk-your-actual-api-key-here"
API_PROVIDER = "openai"
DEFAULT_MODEL_NAME = "text-davinci-003"
```

1. 保存文件

**注意**：`config.py` 文件已被添加到 `.gitignore`，不会被提交到版本控制系统，可以安全地存储你的 API Key。

### 方式二：使用环境变量

#### 智谱AI

**Windows (PowerShell)**

```powershell
$env:ZHIPUAI_API_KEY="your-zhipuai-api-key"
$env:API_PROVIDER="zhipuai"
```

**Windows (CMD)**

```cmd
set ZHIPUAI_API_KEY=your-zhipuai-api-key
set API_PROVIDER=zhipuai
```

**Linux/Mac**

```bash
export ZHIPUAI_API_KEY="your-zhipuai-api-key"
export API_PROVIDER="zhipuai"
```

#### OpenAI

**Windows (PowerShell)**

```powershell
$env:OPENAI_API_KEY="sk-your-actual-api-key-here"
$env:API_PROVIDER="openai"
```

**Windows (CMD)**

```cmd
set OPENAI_API_KEY=sk-your-actual-api-key-here
set API_PROVIDER=openai
```

**Linux/Mac**

```bash
export OPENAI_API_KEY="sk-your-actual-api-key-here"
export API_PROVIDER="openai"
```

### 方式三：命令行参数

在运行脚本时直接传递参数：

**使用智谱AI：**

```bash
python models/logic_program.py --api_provider zhipuai --api_key "your-zhipuai-api-key" --model_name glm-4-flash-250414 --dataset_name ProntoQA
```

**使用OpenAI：**

```bash
python models/logic_program.py --api_provider openai --api_key "sk-..." --model_name text-davinci-003 --dataset_name ProntoQA
```

## 获取 API Key

### 智谱AI API Key

1. 访问 https://open.bigmodel.cn/
2. 登录你的智谱AI账户
3. 在控制台创建API Key
4. 复制生成的 API Key

### OpenAI API Key

1. 访问 https://platform.openai.com/api-keys
2. 登录你的 OpenAI 账户
3. 点击 "Create new secret key"
4. 复制生成的 API Key（注意：API Key 只会显示一次，请妥善保存）

## 使用示例

### 使用智谱AI（默认配置）

配置好 API Key 后，你可以直接运行命令：

```bash
# 使用配置文件中的默认设置（智谱AI）
python models/logic_program.py --dataset_name ProntoQA --split dev

# 指定模型
python models/logic_program.py --dataset_name ProntoQA --split dev --model_name glm-4-flash-250414
```

### 使用OpenAI

```bash
# 使用配置文件中的OpenAI设置
python models/logic_program.py --api_provider openai --dataset_name ProntoQA --split dev --model_name text-davinci-003

# 使用环境变量
# Windows PowerShell:
$env:API_PROVIDER="openai"
$env:OPENAI_API_KEY="sk-..."
python models/logic_program.py --dataset_name ProntoQA --split dev
```

## 支持的模型

### 智谱AI模型

- `glm-4-flash-250414`（推荐，默认）
- `glm-4`
- `glm-3-turbo`
- 其他智谱AI支持的模型

### OpenAI模型

- `text-davinci-003`
- `text-davinci-002`
- `code-davinci-002`
- `gpt-4`
- `gpt-3.5-turbo`

## 配置文件其他选项

`config.py` 还支持设置其他默认值：

- `API_PROVIDER`: API提供商选择（"openai" 或 "zhipuai"）
- `DEFAULT_MODEL_NAME`: 默认使用的模型
- `DEFAULT_MAX_NEW_TOKENS`: 默认最大生成token数
- `DEFAULT_STOP_WORDS`: 默认停止词

这些设置可以通过命令行参数覆盖。

## 安装依赖

使用智谱AI需要安装 `zai` 库：

```bash
pip install zai
```

## 注意事项

1. **API Key安全**：请勿将包含真实API Key的 `config.py` 文件提交到版本控制系统
2. **模型兼容性**：不同API提供商的模型名称不同，请根据选择的提供商使用对应的模型
3. **默认设置**：项目默认使用智谱AI，如需切换请在配置文件中修改 `API_PROVIDER`
