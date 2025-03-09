from typing import List, Tuple, Dict

from jinja2 import FileSystemLoader, Environment, StrictUndefined, TemplateNotFound


class PromptTemplate:
    """Prompt Template
    """

    def get(self, prompt_name: str, **kwargs) -> str:
        raise NotImplementedError

    def list_templates(self) -> List[str]:
        raise NotImplementedError


class JinjaPromptTemplate(PromptTemplate):
    """Jinja Prompt Template
    """

    def __init__(self, paths: List[str]):
        self._template_env = Environment(
            loader=FileSystemLoader(paths),
            autoescape=False,
            lstrip_blocks=True,
            trim_blocks=True,
            keep_trailing_newline=True,
            # undefined=StrictUndefined  # TODO：最终使用 StrictUndefined
        )

    def get(self, prompt_name: str, **kwargs) -> str:
        try:
            tpl = self._template_env.get_template(prompt_name)
        except TemplateNotFound as err:
            raise ValueError(f"模板不存在: {prompt_name}") from err
        return tpl.render(**kwargs)

    def list_templates(self) -> List[str]:
        return self._template_env.list_templates()


class PromptLoader:
    """PromptLoader
    """

    _prompt_templates: List[PromptTemplate] = None
    _prompts: Dict[Tuple[str, ...], PromptTemplate] = None

    @classmethod
    def from_paths(cls, paths: List[str]):
        """load Jinja Prompt Template from paths"""
        cls._prompt_templates = [JinjaPromptTemplate(paths)]
        return cls

    @classmethod
    def get_prompt(cls, prompt_name: str, **kwargs) -> str:
        """get prompt by prompt name"""
        for temp_tpl in cls._prompt_templates:
            if prompt_name in temp_tpl.list_templates():
                return temp_tpl.get(prompt_name, **kwargs)
        raise ValueError(f"prompt template not found: {prompt_name}")


__all__ = ["PromptLoader"]
