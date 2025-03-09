import inspect
from typing import Any, Callable, Dict, Optional, Literal, Annotated, get_args, get_origin


def get_function_schema(f: Callable[..., Any], *, name: Optional[str] = None, description: Optional[str] = None) -> Dict[str, Any]:
    """Get the JSON schema of a function defined in OpenAI API

    Args:
        f (Callable[..., Any]): The function to get the schema of.
        name (Optional[str], optional): The name of the function. Defaults to f.__name__.
        description (Optional[str], optional): The description of the function. Defaults to f.__doc__.

    Returns:
        Dict[str, Any]: The JSON schema of the function.

    Raises:
        AssertionError: If the function does not have a description.
    """

    type_map = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object",
        type(None): "null"
    }

    # function name
    if name is None:
        name = f.__name__

    # function description
    if description is None:
        if f.__doc__ is None:
            raise AssertionError(f"function {name} needs a description or docstring")
        description = f.__doc__

    # function parameters
    signature = inspect.signature(f)
    parameters = {
        "type": "object",
        "properties": {},
        "required": []
    }
    for param in signature.parameters.values():
        _param = {}
        # parameter default value
        if param.default == inspect.Signature.empty:
            parameters["required"].append(param.name)
        else:
            _param["default"] = param.default

        # parameter description
        if param.annotation == inspect.Signature.empty:
            raise AssertionError(f"function {name} parameter {param.name} needs an annotation")
        if get_origin(param.annotation) is not Annotated:
            raise AssertionError(f"function {name} parameter {param.name} annotation should be Annotated")
        param_type, _param["description"] = get_args(param.annotation)
        
        # parameter type
        if get_origin(param_type) is None:
            # base type
            _param["type"] = type_map[param_type]
        else:
            # TODO: wapper type, e.g. Literal
            wapper_type = get_origin(param_type)
            if wapper_type is Literal:
                _param["enum"] = list(get_args(param_type))
                _param["type"] = type_map[type(_param["enum"][0])]

        parameters["properties"][param.name] = _param

    return {
        "type": "function",
        "function": {
            "description": description,
            "name": name,
            "parameters": parameters,
        }
    }