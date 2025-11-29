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

# ========== iflow API配置 ==========
# iflow API Key
# iflow 是一个兼容 OpenAI API 的服务
IFLOW_API_KEY = "sk-64e5a7718cd09298a70a92578c89a50e"
#sk-55fa9e3d00b983bff51f62a1b61dfe1d
#sk-6f66c56e611e6945493655d7418d3453
#sk-64e5a7718cd09298a70a92578c89a50e
#sk-4879f1cc4df9aa4e5280e82e085e7d72
#sk-3878b65faa7cc385bc8abafc8de74e85

IFLOW_BASE_URL = "https://apis.iflow.cn/v1"  # iflow API 的 base_url

# ========== 默认设置 ==========
# API提供商选择: "openai"、"zhipuai" 或 "iflow"
API_PROVIDER = "iflow"  # 默认使用 iflow

# 默认模型设置（可选）
# OpenAI模型: text-davinci-003, gpt-4, gpt-3.5-turbo
# 智谱AI模型: glm-4-flash-250414, glm-4, glm-3-turbo 等
# iflow模型: TBStars2-200B-A13B
DEFAULT_MODEL_NAME = "TBStars2-200B-A13B"  # iflow 默认模型

# 其他默认设置（可选）
DEFAULT_MAX_NEW_TOKENS = 1024
DEFAULT_STOP_WORDS = "------"

