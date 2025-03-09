from typing import Any

from core.llms.base_client import (
    BaseLLMClient,
    LLMClientResponse,
    LLMResponseChoice,
    LLMResponseMessage,
    LLMResponseMessageToolCall,
    LLMResponseMessageToolCallFunction,
    LLMResponseUsage
)

try:
    import anthropic
    from anthropic.types.message import Message as AnthropicChatCompletion
except ImportError or ModuleNotFoundError:
    raise ImportError("please install Anthropic Python SDK：`pip install anthropic`")

from core.logger.runtime import get_logger

class AnthropicClient(BaseLLMClient):
    BASE_URL = "https://api.anthropic.com/v1"

    def __init__(self, llm_config, **kwargs):
        logger = get_logger(name="Anthropic Client", filename="anthropic_client.log")
        super().__init__(llm_config, logger=logger, **kwargs)
        self._client = anthropic.Anthropic(
            base_url=self._base_url, api_key=self._api_key, timeout=self._timeout
        )

    def do_create(self, messages, generate_config) -> LLMClientResponse:
        # 目前不允许流式输出
        generate_config["stream"] = False

        try:
            response = self._client.messages.create(
                model=self._model, messages=messages, **generate_config
            )
        except Exception as e:
            self.logger.error(f"OpenAI Client Error：{e}")
            raise e

        return self.cast_response(response)

    @classmethod
    def cast_response(cls, response: AnthropicChatCompletion) -> LLMClientResponse:
        usage = LLMResponseUsage(
            prompt_tokens=response.usage.input_tokens,
            completion_tokens=response.usage.output_tokens,
            total_tokens=response.usage.input_tokens + response.usage.output_tokens,
        )
        choices = [
            LLMResponseChoice(
                message=LLMResponseMessage(
                    content=choice.text,
                    role=response.role,
                    tool_calls=[
                        LLMResponseMessageToolCall(
                            id=choice.id,
                            type="function",
                            function=LLMResponseMessageToolCallFunction(
                                name=choice.name,
                                arguments=str(choice.input),
                            ),
                        )
                    ]
                    if choice.type == "tool_use"
                    else None,
                ),
                logprobs=None,
                finish_reason={
                    "end_turn": "stop",
                    "max_tokens": "length",
                    "tool_use": "tool_calls",
                    "stop_sequence": "content_filter",
                }.get(response.stop_reason, "stop") if response.stop_reason else None,
            )
            for choice in response.content
        ]
        return LLMClientResponse(choices=choices, model=response.model, usage=usage)
