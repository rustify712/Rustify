import json
import re
import copy
from enum import Enum
from typing import Callable, Optional

import shortuuid

from core.logger.base_logger import BaseLogger
from core.utils.prompt_loader import PromptLoader
from core.llms.base_client import Tool, LLMClientResponse, LLMResponseMessageToolCall
from core.llms.llm_factory import LLMFactory
from core.logger.runtime import get_logger
from core.utils.func_utils import get_function_schema



class AgentResponseStatus(Enum):
    DONE = "done"
    ERROR = "error"
    RETRY = "retry"


class AgentResponseType:
    """
    智能体的响应类型
    """
    CHAT = "chat"
    TOOL = "tool"


class AgentRequest:
    """
    智能体的请求体
    """

    def __init__(self, type: str, data: Optional[dict] = None, origin_agent: Optional["BaseAgent"] = None):
        self.type = type
        self.data = data or {}
        self.origin_agent = origin_agent


class AgentResponse:
    """
    智能体的响应体

    Attributes:
        status: AgentResponseStatus 响应状态
        type: AgentResponseType 响应类型
        agent: BaseAgent 智能体
        data: Optional[dict] 数据
        error: Optional[dict] 错误
    """

    def __init__(
            self,
            status: AgentResponseStatus,
            rtype: str,
            agent: "BaseAgent",
            data: Optional[dict] = None,
            messages: Optional[list[dict]] = None
    ):
        self.status = status
        self.type = rtype
        self.agent = agent
        self.data = data or {}
        self.messages = messages or []

    @staticmethod
    def done(agent: "BaseAgent", rtype: str, data: Optional[dict] = None) -> "AgentResponse":
        return AgentResponse(status=AgentResponseStatus.DONE, rtype=rtype, agent=agent, data=data)

    @staticmethod
    def error(agent: "BaseAgent", rtype: str, error: dict) -> "AgentResponse":
        return AgentResponse(status=AgentResponseStatus.ERROR, rtype=rtype, agent=agent, data=error)

    def set_messages(self, messages: list[dict]):
        self.messages = messages
        return self

    def __str__(self):
        return f"AgentResponse(status={self.status}, type={self.type}, agent={self.agent}, data={self.data})"

    def __repr__(self):
        return self.__str__()


class ToolCallResult:
    """
    工具调用结果
    """

    success: bool
    """是否成功"""
    tool_call_id: str
    """工具调用 ID"""
    name: str
    """工具调用名称"""
    arguments: dict
    """工具调用参数"""
    content: str
    """工具调用结果或报错"""

    def __init__(self, success: bool, tool_call_id: str, name: str, arguments: dict, content: str):
        self.success = success
        self.id = tool_call_id
        self.name = name
        self.arguments = arguments
        self.content = content

    @staticmethod
    def ok(tool_call_id: str, name: str, arguments: dict, content: str):
        return ToolCallResult(success=True, tool_call_id=tool_call_id, name=name, arguments=arguments, content=content)

    @staticmethod
    def error(tool_call_id: str, name: str, arguments: dict, content: str):
        return ToolCallResult(success=False, tool_call_id=tool_call_id, name=name, arguments=arguments, content=content)


class BaseAgent:
    """
    智能体基类
    """

    ROLE = "base"
    """角色"""
    DESCRIPTION = "A powerful and efficient AI assistant that completes tasks accurately and resourcefully."
    """智能体介绍"""

    JSON_PATTERN = re.compile(r"```json\s*(.*?)\s*```", re.DOTALL)

    def __init__(
            self,
            llm_config: dict,
            *,
            name: Optional[str] = None,
            logfile: Optional[str] = None,
            logger: Optional[BaseLogger] = None
    ):
        self.name = name or f"{self.ROLE}_{shortuuid.uuid()[:8]}"
        self.llm_config = llm_config
        self.llm_client = LLMFactory.create_llm_from_config(llm_config)
        if logger:
            self.logger = logger
        else:
            logfile = logfile or f"{self.name}.log"
            self.logger = get_logger(name=self.name, filename=logfile)

        self.tools = []
        self.tool_map = {}
        self.tool_reply_factory_map = {}

    @property
    def system_message(self):
        return PromptLoader.get_prompt(
            f"{self.ROLE}/system.prompt",
            tools=self.tools
        )

    def run(self, pre_response: AgentResponse) -> AgentResponse:
        """
        运行智能体

        Args:
            pre_response: AgentResponse 前一个消息的响应体

        Returns:
            AgentResponse 响应体
        """
        raise NotImplementedError

    def _validate_messages(self, messages: list[dict]) -> list[dict]:
        """验证消息的合法性，返回合法的消息

        Args:
            messages: list[dict] 消息

        Returns:
            list[dict]: 合法的消息
        """
        allow_keys = {
            "system": ["role", "content", "name"],
            "user": ["role", "content", "name"],
            "assistant": ["role", "content", "name", "tool_calls"],
            "tool": ["role", "content", "name", "tool_call_id"]
        }
        for message in messages:
            if "role" not in message:
                raise ValueError("Role is required")
            if message["role"] not in allow_keys:
                raise ValueError(f"Invalid role: {message['role']}")
            for key in message.keys():
                if key not in allow_keys[message["role"]]:
                    del message[key]
        if not messages:
            raise ValueError("Messages cannot be empty")
        if messages[0]["role"] != "system":
            messages.insert(0, {"role": "system", "content": self.system_message})

        return messages

    def call_llm(self, messages: list[dict], **kwargs) -> LLMClientResponse:
        """调用大模型并返回结果

        由于部分大模型目前的工具能力参差不齐，这里不使用 API 工具调用，转而解析 JSON
        """
        messages = copy.deepcopy(messages)
        # 验证并返回合法消息
        messages = self._validate_messages(messages)
        # 使用 JSON 来调用工具，而非 API Tool Call
        for message in messages:
            # 去除 tool_calls
            if "tool_calls" in message:
                del message["tool_calls"]
            # 角色为 tool 的消息改为 user
            if message.get("role") == "tool":
                message["role"] = "user"
                # 去除 tool_call_id
                if "tool_call_id" in message:
                    del message["tool_call_id"]
        response = self.llm_client.create(messages=messages, **kwargs)
        # 虽然这里对 n != 1 的情况做了处理，但目前实际上只有允许 n = 1 的情况
        for response_choice in response.choices:
            response_content = response_choice.message.content
            # 提取返回消息中的所有 JSON
            json_blocks = self.JSON_PATTERN.findall(response_content)
            if not json_blocks:
                continue
            # 解析 JSON
            tool_calls = []
            for json_block in json_blocks:
                # 检查是否是工具调用的 JSON 消息
                json_block = json_block.strip(" \n\r\t`")
                try:
                    json_obj = json.loads(json_block)
                    tool_name = json_obj.get("tool_name", None)
                    tool_input = json_obj.get("tool_input", None)
                    if not tool_name or not tool_input:
                        # 非工具调用
                        continue
                    if tool_name not in self.tool_map:
                        self.logger.warning(f"Tool not found: {tool_name}")
                        continue
                    tool_calls.append({
                        "id": shortuuid.uuid(),
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "arguments": json.dumps(tool_input)
                        }
                    })
                except json.JSONDecodeError:
                    continue
            if not tool_calls:
                continue
            # 将工具调用添加到 response 中
            response_choice.message.tool_calls = tool_calls
        return response

    def call_tools(self, tool_calls: list[LLMResponseMessageToolCall]) -> list[ToolCallResult]:
        """调用工具

        Args:
            tool_calls: list[LLMResponseMessageToolCall] 工具调用列表

        Returns:
            list[ToolCallResult]: 工具调用结果列表
        """
        if len(tool_calls) == 0:
            return []
        tool_results = []
        for tool_call in tool_calls:
            tool = self.tool_map.get(tool_call.function.name)
            tool_args = json.loads(tool_call.function.arguments)
            if not tool:
                self.logger.warning(f"Tool not found: {tool_call.function.name}")
                continue
            # TODO: 添加 tool reply factory, 用于支持自定义解析工具调用结果
            try:
                tool_return = tool(**tool_args)
                tool_result = ToolCallResult.ok(
                    tool_call.id,
                    tool_call.function.name,
                    tool_args,
                    tool_return
                )
            except Exception as e:
                tool_result = ToolCallResult.error(
                    tool_call.id,
                    tool_call.function.name,
                    tool_args,
                    str(e)
                )
            tool_results.append(tool_result)
        return tool_results

    def call(self, messages: list[dict], max_rounds: int = 10, enable_tools: bool = True, **kwargs) -> AgentResponse:
        """调用智能体

        Args:
            messages: list[dict] 消息列表
            max_rounds: int 最大轮数
            enable_tools: bool 是否启用工具
            **kwargs: dict 其他参数
        """
        cur_round = 0
        chat_messages = copy.deepcopy(messages)
        init_messages_len = len(messages)
        while cur_round < max_rounds:
            response = self.call_llm(messages, **kwargs)
            chat_messages.append(response.choices[0].message.model_dump())
            if enable_tools:
                if response.choices[0].message.tool_calls:
                    tool_results = self.call_tools(response.choices[0].message.tool_calls)
                    tool_messages = self.convert_tool_call_results(tool_results)
                    chat_messages.extend(tool_messages)
            if chat_messages[-1]["role"] == "assistant":
                return AgentResponse.done(self, AgentResponseType.CHAT).set_messages(chat_messages[init_messages_len:])
        return AgentResponse.done(self, AgentResponseType.CHAT).set_messages(chat_messages[init_messages_len:])

    @staticmethod
    def convert_tool_call_results(tool_results: list[ToolCallResult]) -> list[dict]:
        """转换工具调用结果"""
        return [
            {
                "role": "tool",
                "tool_call_id": tool_result.id,
                "content": tool_result.content
            }
            for tool_result in tool_results
        ]

    def register_tool(
            self,
            function: Callable,
            *,
            name: Optional[str] = None,
            description: Optional[str] = None,
            reply_factory: Optional[Callable] = None,
    ):
        tool_desc = get_function_schema(function, name=name, description=description)
        tool = Tool(**tool_desc)
        self.tools.append(tool)
        self.tool_map[tool.function.name] = tool
        if reply_factory:
            self.tool_reply_factory_map[tool.function.name] = reply_factory
