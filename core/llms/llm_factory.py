from typing import Dict, Any

class LLMFactory:

    @staticmethod
    def create_llm_from_config(llm_config: Dict[str, Any], **kwargs):
        """create a large language model client from config

        Args:
            llm_config (Dict[str, Any]): llm config
        """
        if "provider" not in llm_config:
            raise ValueError("llm_config missing provider field")
        llm_client_type = llm_config["provider"]

        if llm_client_type == "openai":
            from core.llms.openai_client import OpenAIClient
            return OpenAIClient(llm_config, **kwargs)
        elif llm_client_type == "anthropic":
            from core.llms.anthropic_client import AnthropicClient
            return AnthropicClient(llm_config, **kwargs)
        else:
            # 需要自行实现或使用 one-api 等工具
            raise ValueError(f"unsupported large language model type: {llm_client_type}")
