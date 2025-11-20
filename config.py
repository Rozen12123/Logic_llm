# API配置文件
# 请在此处设置你的API Key
# 注意：请不要将此文件提交到版本控制系统

# ========== OpenAI API配置 ==========
# OpenAI API Key
# 获取方式：https://platform.openai.com/api-keys
OPENAI_API_KEY = "your-api-key-here"

# ========== 智谱AI (ZhipuAI) API配置 ==========
# 智谱AI API Key
# 获取方式：https://open.bigmodel.cn/
ZHIPUAI_API_KEY = "78443ce13bdd4f72809efda1abd95af4.s6TNCSWw3Q4Ci2wx"

# ========== 默认设置 ==========
# API提供商选择: "openai" 或 "zhipuai"
API_PROVIDER = "zhipuai"  # 默认使用智谱AI

# 默认模型设置（可选）
# OpenAI模型: text-davinci-003, gpt-4, gpt-3.5-turbo
# 智谱AI模型: glm-4-flash-250414, glm-4, glm-3-turbo 等
DEFAULT_MODEL_NAME = "glm-4-flash-250414"  # 智谱AI默认模型

# 其他默认设置（可选）
DEFAULT_MAX_NEW_TOKENS = 1024
DEFAULT_STOP_WORDS = "------"

