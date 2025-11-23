"""
配置加载工具
用于从配置文件或环境变量中读取API Key等配置信息
支持OpenAI、智谱AI (ZhipuAI) 和 iflow
"""
import os

def load_api_key(provider="zhipuai"):
    """
    加载API Key（支持OpenAI、智谱AI和iflow）
    优先级：命令行参数 > 环境变量 > 配置文件
    
    Args:
        provider: API提供商，"openai"、"zhipuai" 或 "iflow"
    
    Returns:
        str: API Key，如果都未设置则返回None
    """
    if provider == "zhipuai":
        # 首先尝试从环境变量读取智谱AI API Key
        api_key = os.getenv('ZHIPUAI_API_KEY')
        
        # 如果环境变量没有，尝试从配置文件读取
        if not api_key:
            try:
                import config
                api_key = getattr(config, 'ZHIPUAI_API_KEY', None)
                # 如果配置文件中是默认值，则返回None
                if api_key and api_key != "your-zhipuai-api-key-here":
                    return api_key
            except ImportError:
                # 配置文件不存在，返回None
                pass
    elif provider == "iflow":
        # 首先尝试从环境变量读取 iflow API Key
        api_key = os.getenv('IFLOW_API_KEY')
        
        # 如果环境变量没有，尝试从配置文件读取
        if not api_key:
            try:
                import config
                api_key = getattr(config, 'IFLOW_API_KEY', None)
                # 如果配置文件中是默认值，则返回None
                if api_key and api_key != "your-iflow-api-key-here":
                    return api_key
            except ImportError:
                # 配置文件不存在，返回None
                pass
    else:  # openai
        # 首先尝试从环境变量读取OpenAI API Key
        api_key = os.getenv('OPENAI_API_KEY')
        
        # 如果环境变量没有，尝试从配置文件读取
        if not api_key:
            try:
                import config
                api_key = getattr(config, 'OPENAI_API_KEY', None)
                # 如果配置文件中是默认值，则返回None
                if api_key and api_key != "your-api-key-here":
                    return api_key
            except ImportError:
                # 配置文件不存在，返回None
                pass
    
    return api_key

def load_base_url(provider="zhipuai"):
    """
    加载 base_url（用于 iflow 等兼容 OpenAI 的 API）
    优先级：环境变量 > 配置文件 > 默认值
    
    Args:
        provider: API提供商，"openai"、"zhipuai" 或 "iflow"
    
    Returns:
        str: base_url，如果未设置则返回None（使用默认值）
    """
    if provider == "iflow":
        # iflow 的默认 base_url
        base_url = os.getenv('IFLOW_BASE_URL', 'https://apis.iflow.cn/v1')
        
        # 如果环境变量没有，尝试从配置文件读取
        if base_url == 'https://apis.iflow.cn/v1':
            try:
                import config
                base_url = getattr(config, 'IFLOW_BASE_URL', 'https://apis.iflow.cn/v1')
            except ImportError:
                pass
        return base_url
    elif provider == "openai":
        # OpenAI 也支持自定义 base_url
        base_url = os.getenv('OPENAI_BASE_URL')
        if not base_url:
            try:
                import config
                base_url = getattr(config, 'OPENAI_BASE_URL', None)
            except ImportError:
                pass
        return base_url
    else:
        return None

def load_api_provider():
    """
    加载API提供商配置
    优先级：环境变量 > 配置文件 > 默认值（zhipuai）
    
    Returns:
        str: "openai"、"zhipuai" 或 "iflow"
    """
    # 首先尝试从环境变量读取
    provider = os.getenv('API_PROVIDER', '').lower()
    
    # 如果环境变量没有，尝试从配置文件读取
    if not provider:
        try:
            import config
            provider = getattr(config, 'API_PROVIDER', 'zhipuai')
        except ImportError:
            provider = 'zhipuai'
    
    # 确保返回值是有效的
    if provider not in ['openai', 'zhipuai', 'iflow']:
        provider = 'zhipuai'
    
    return provider

def load_config():
    """
    加载所有配置
    
    Returns:
        dict: 包含所有配置的字典
    """
    # 首先确定使用哪个API提供商
    api_provider = load_api_provider()
    
    config_dict = {
        'api_provider': api_provider,
        'api_key': load_api_key(api_provider),
        'base_url': load_base_url(api_provider),
        'model_name': None,
        'max_new_tokens': None,
        'stop_words': None
    }
    
    # 尝试从配置文件读取其他配置
    try:
        import config
        config_dict['model_name'] = getattr(config, 'DEFAULT_MODEL_NAME', None)
        config_dict['max_new_tokens'] = getattr(config, 'DEFAULT_MAX_NEW_TOKENS', None)
        config_dict['stop_words'] = getattr(config, 'DEFAULT_STOP_WORDS', None)
    except ImportError:
        pass
    
    return config_dict

