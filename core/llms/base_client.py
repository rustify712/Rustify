import json
from abc import abstractmethod
import copy
import re
from typing import Any, Dict, Generic, List, Literal, Optional, TypeVar, Type

import jsonref
from pydantic import BaseModel


class ToolFunction(BaseModel):
    name: str
    description: str
    parameters: Dict[str, Any]


class Tool(BaseModel):
    type: Literal["function"] = "function"
    function: ToolFunction


class LLMResponseMessageToolCallFunction(BaseModel):
    name: str
    arguments: Dict[str, Any] | str


class LLMResponseMessageToolCall(BaseModel):
    id: str
    type: str
    function: LLMResponseMessageToolCallFunction


class LLMResponseMessage(BaseModel):
    content: Optional[str]
    role: str
    tool_calls: Optional[List[LLMResponseMessageToolCall]]


class LLMResponseChoice(BaseModel):
    message: LLMResponseMessage
    logprobs: Optional[Any]
    finish_reason: Optional[str]


class LLMResponseUsage(BaseModel):
    completion_tokens: int
    prompt_tokens: int
    total_tokens: int


JSON_FORMAT_T = TypeVar('JSON_FORMAT_T', bound=Type[BaseModel])


class LLMClientResponse(BaseModel, Generic[JSON_FORMAT_T]):
    choices: List[LLMResponseChoice]
    model: str
    usage: Optional[LLMResponseUsage]

    _obj_type: Optional[JSON_FORMAT_T] = None

    @property
    def format_object(self) -> Optional[JSON_FORMAT_T]:
        if self._obj_type is None:
            return None
        try:
            return parse_json_format(self.choices[0].message.content, self._obj_type)
        except Exception as e:
            print(str(e))
            return None

ALLOW_GENERATE_CONFIG_KEYS = [
    "max_tokens",
    "frequency_penalty",
    "presence_penalty",
    "stop",
    "temperature",
    "top_p",
    "tools",
    "tool_choice",
    "logprobs",
]


def construct_generate_config(llm_config: dict):
    """Filter out unsupported configuration items
    
    Args:
        llm_config (dict): the llm config
    """
    # TODO: support more keys
    allow_keys = ALLOW_GENERATE_CONFIG_KEYS
    generate_config = {}
    for key in allow_keys:
        if key in llm_config:
            generate_config[key] = llm_config[key]
    return generate_config


def merge_generate_config(generate_config: dict, new_config: dict):
    """merge new config into generate config
    
    Args:
        generate_config (dict): the generate config
        new_config (dict): the new config
    """
    allow_keys = ALLOW_GENERATE_CONFIG_KEYS
    for key in allow_keys:
        if key in new_config:
            generate_config[key] = new_config[key]
    enable_tool_call = "tools" in generate_config and generate_config["tools"]
    if not enable_tool_call:
        # remove tool_choice and tools if tools is not enabled
        if "tool_choice" in generate_config:
            del generate_config["tool_choice"]
        if "tools" in generate_config:
            del generate_config["tools"]


def remove_json_defs(d, visited=None):
    if visited is None:
        visited = set()
    obj_id = id(d)
    if obj_id in visited:
        return None
    visited.add(obj_id)
    if isinstance(d, dict):
        return {k: remove_json_defs(v, visited) for k, v in d.items() if k != "$defs"}
    elif isinstance(d, list):
        return [remove_json_defs(v, visited) for v in d]
    else:
        return d


md_json_pattern = re.compile(r"```(json)?(.*)", re.DOTALL)


def parse_json_format(data: str, json_format: JSON_FORMAT_T) -> JSON_FORMAT_T:
    # 解析 Markdown 代码块或文本格式的 JSON
    try:
        if data.strip().startswith("```json"):
            match = md_json_pattern.search(data)
            if match:
                data = match.group(2)
                data = data.strip(" \n\r\t`")
        return json_format.model_validate_json(data)
    except Exception as e:
        raise f"parse json object error: {e}, json_format: {json_format}"


class BaseLLMClient:
    """Base class for LLM clients.

    Args:
        llm_config (dict): The configuration for the LLM client.
    """

    BASE_URL: str = None
    DEFAULT_TIMEOUT_SECONDS: int = 300

    DEFAULT_JSON_SCHEMA_PROMPT = """IMPORTANT: Your response MUST conform to this JSON schema:```json\n{json_schema}\n```\nYOU MUST NEVER add any additional fields to your response, and NEVER add additional preamble like "Here is your JSON"."""

    def __init__(self, llm_config: dict, logger=None, **kwargs):
        if "base_url" not in llm_config and self.BASE_URL is None:
            raise ValueError(f"`base_url` must be specified in llm_config or in the {self.__class__.__name__} class")
        self._base_url = llm_config.get("base_url", self.BASE_URL)
        if "api_key" not in llm_config:
            raise ValueError("`api_key` must be specified in llm_config")
        self._api_key = llm_config.get("api_key")
        if "model" not in llm_config:
            raise ValueError("`model` must be specified in llm_config")
        self._model = llm_config.get("model")
        self._timeout = llm_config.get("timeout", self.DEFAULT_TIMEOUT_SECONDS)

        self._generate_config = construct_generate_config(llm_config)
        self.logger = logger

    def create(self, messages: list[dict], *, json_format: JSON_FORMAT_T = None, **kwargs) -> LLMClientResponse[
        JSON_FORMAT_T]:
        """创建一个大模型调用请求

        Args:
            messages: 一个消息列表，每个消息是一个字典，包含 `role` 和 `content` 等字段
            json_format: 用于解析 JSON 格式的 Pydantic 模型
            **kwargs: 生成配置

        Returns:
            LLMClientResponse: 大模型调用的响应
        """
        # construct generate config
        generate_config = copy.deepcopy(self._generate_config)
        merge_generate_config(generate_config, kwargs)

        # request
        if json_format:
            json_message = self._construct_json_format(json_format, messages, generate_config)
            response = self.do_create(messages + [json_message], generate_config)
            response._obj_type = json_format
        else:
            response = self.do_create(messages, generate_config)

        if self.logger:
            self.logger.debug(f"llm call: {messages} -> {response}")
        else:
            print(f"llm call: {messages} -> {response}")
        # TODO: 利用小模型将文本转换为 JSON 格式，而非直接生成 JSON。（ Cot的情况下要比直接生成 JSON 的回答结果好得多 ）
        return response

    @abstractmethod
    def do_create(self, messages: list[dict], generate_config: dict) -> LLMClientResponse:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def cast_response(cls, response: Any) -> LLMClientResponse:
        raise NotImplementedError

    def _construct_json_format(self, json_format, messages: list[dict], generate_config: dict) -> dict:
        if issubclass(json_format, BaseModel):
            # TODO: 解决 BaseModel 递归引用的问题
            # json_schema = remove_json_defs(jsonref.loads(json.dumps(json_format.model_json_schema())))
            json_schema = json_format.model_json_schema()
            json_format_message = self.DEFAULT_JSON_SCHEMA_PROMPT.format(json_schema=json_schema)
            return {
                "role": "user",
                "content": json_format_message
            }
        raise ValueError("json_format 必须是 Pydantic 模型")
