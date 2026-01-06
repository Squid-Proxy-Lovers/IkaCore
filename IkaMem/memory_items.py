from typing import Any, Optional


class STMemItem:
    """short-term memory data container."""
    def __init__(
        self,
        data: Any,
        agent: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ):
        """
        init short-term memory item.
        
        Args:
            data: the actual data/content to store
            agent: optional agent identifier
            metadata: optional metadata dictionary
        """
        self.data = data
        self.agent = agent
        self.metadata = metadata if metadata is not None else {}


class LTMemItem:
    """long-term memory data container."""
    def __init__(
        self,
        agent: str,
        task: str,
        expected_output: str,
        datetime: str,
        quality: Optional[float] = None,
        metadata: Optional[dict[str, Any]] = None,
    ):
        """
        init long-term memory item.
        
        Args:
            agent: agent identifier
            task: task description
            expected_output: expected output description
            datetime: timestamp of the memory
            quality: optional quality score
            metadata: optional metadata dictionary
        """
        self.task = task
        self.agent = agent
        self.quality = quality
        self.datetime = datetime
        self.expected_output = expected_output
        self.metadata = metadata if metadata is not None else {}

