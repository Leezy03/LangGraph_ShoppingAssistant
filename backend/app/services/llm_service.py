"""LLM服务模块"""

import os
from typing import Optional

from langchain_openai import ChatOpenAI

from ..config import get_settings

# 全局LLM实例
_llm_instance = None


def _get_env(name: str, fallback: Optional[str] = None) -> Optional[str]:
    value = os.getenv(name)
    return value if value not in (None, "") else fallback


def get_llm() -> ChatOpenAI:
    """
    获取 LangChain ChatModel 实例(单例模式)。

    环境变量继续兼容原项目的 LLM_* 命名,同时兼容 OPENAI_* 命名。
    """
    global _llm_instance

    if _llm_instance is None:
        config = get_llm_config()

        _llm_instance = ChatOpenAI(
            model=config["model"],
            api_key=config["api_key"],
            base_url=config["base_url"],
            timeout=config["timeout"],
            temperature=0,
        )

        print("✅ LLM服务初始化成功")
        print("   运行时: LangChain ChatOpenAI")
        print(f"   模型: {config['model']}")
        print(f"   Base URL: {config['base_url']}")

    return _llm_instance


def get_llm_config() -> dict:
    """获取 OpenAI-compatible LLM 配置。"""
    settings = get_settings()
    api_key = (
        _get_env("LLM_API_KEY")
        or _get_env("OPENAI_API_KEY")
        or settings.openai_api_key
    )
    base_url = (
        _get_env("LLM_BASE_URL")
        or _get_env("OPENAI_BASE_URL")
        or settings.openai_base_url
    )
    model = (
        _get_env("LLM_MODEL_ID")
        or _get_env("OPENAI_MODEL")
        or settings.openai_model
    )
    timeout = float(_get_env("LLM_TIMEOUT", "60") or "60")
    return {
        "api_key": api_key,
        "base_url": base_url,
        "model": model,
        "timeout": timeout,
    }


def reset_llm():
    """重置LLM实例(用于测试或重新配置)"""
    global _llm_instance
    _llm_instance = None
