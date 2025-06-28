from core.agents.base import BaseAgent

class Reasoner(BaseAgent):
    """推理智能体，极其擅长推理
    """

    ROLE = "reasoner"
    DESCRIPTION = "A powerful agent that is good at reasoning."

    def __init__(self, llm_config: dict, **kwargs):
        super().__init__(llm_config, **kwargs)
