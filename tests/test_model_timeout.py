from eyetools.core.registry import ToolRegistry
from eyetools.core.tool_manager import ToolManager


def test_tool_manager_uses_15_minute_model_timeout():
    tool_manager = ToolManager(ToolRegistry())

    assert tool_manager._model_timeout_s == 900.0
    assert tool_manager._process_manager.default_timeout_s == 900.0
