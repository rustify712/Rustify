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
    import openai
    from openai.types.chat.chat_completion import ChatCompletion as OpenAIChatCompletion
except ImportError or ModuleNotFoundError:
    raise ImportError("please install OpenAI Python SDK：`pip install openai`")

from core.logger.runtime import get_logger



class OpenAIClient(BaseLLMClient):
    BASE_URL = "https://api.openai.com/v1"

    def __init__(self, llm_config, **kwargs):
        logger = get_logger(name="OpenAI Client", filename="openai_client.log")
        super().__init__(llm_config, logger=logger, **kwargs)
        self._client = openai.Client(
            base_url=self._base_url, api_key=self._api_key, timeout=self._timeout
        )

    def do_create(self, messages, generate_config) -> LLMClientResponse:
        # 目前不允许流式输出
        generate_config["stream"] = False

        try:
            response = self._client.chat.completions.create(
                model=self._model, messages=messages, **generate_config
            )
        except Exception as e:
            self.logger.error(f"OpenAI Client Error：{e}")
            raise e

        return self.cast_response(response)

    @classmethod
    def cast_response(cls, response: OpenAIChatCompletion) -> LLMClientResponse:
        usage = LLMResponseUsage(
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            total_tokens=response.usage.total_tokens,
        )
        choices = [
            LLMResponseChoice(
                message=LLMResponseMessage(
                    content=choice.message.content,
                    role=choice.message.role,
                    tool_calls=[
                        LLMResponseMessageToolCall(
                            id=tool_call.id,
                            type=tool_call.type,
                            function=LLMResponseMessageToolCallFunction(
                                name=tool_call.function.name,
                                arguments=tool_call.function.arguments,
                            ),
                        )
                        for tool_call in choice.message.tool_calls
                    ]
                    if choice.message.tool_calls
                    else None,
                ),
                logprobs=choice.logprobs,
                finish_reason=choice.finish_reason,
            )
            for choice in response.choices
        ]
        return LLMClientResponse(choices=choices, model=response.model, usage=usage)
