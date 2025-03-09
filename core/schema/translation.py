from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class TranslationUnitNode(BaseModel):
    filepath: str = Field(description="转译节点所在文件路径")
    id: str = Field(description="节点 ID")
    name: str = Field(description="节点名称")
    type: str = Field(description="节点类型")
    text: str = Field(description="节点代码")
    description: Optional[str] = Field(description="节点描述", default="")


class TranslationTaskSource(BaseModel):
    id: str = Field(description="待转译的 C/C++ 节点 ID")
    name: str = Field(description="待转译的 C/C++ 节点名称")
    nodes: List[TranslationUnitNode] = Field(description="待转译的 C/C++ 节点")
    # related_node_ids: List[str] = Field(description="相关节点 ID")
    description: Optional[str] = Field(description="节点描述", default="")


class TranslationTaskStatus:
    INIT = "init"
    """初始化"""
    RUNNING = "running"
    """转译中"""
    COMPLETION = "completion"
    """转译代码完成"""
    DONE = "done"
    """转译任务完成"""
    FAILED = "failed"
    """转译任务失败"""


class TranslationTask(BaseModel):
    source: TranslationTaskSource = Field(description="待转译的 C/C++ 节点")
    target: Optional[TranslationUnitNode] = Field(description="转译后的 Rust 节点")
    status: Literal["init", "running", "completion", "done", "failed"] = Field(default=False, description="是否已经转译")
    prerequisites: List[str] = Field(description="转译任务的前置任务的ID", default=[])


    def __hash__(self):
        return hash(self.source.id)
