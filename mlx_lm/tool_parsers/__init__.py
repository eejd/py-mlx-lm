# Copyright © 2024 Apple Inc.


class NotAToolCall(ValueError):
    """Raised by a parser for tool-call text that is not a call but content.

    The server returns ``content`` as part of the message text. Any other ``ValueError``
    means the call text was cut short, and the call is dropped.
    """

    def __init__(self, content: str):
        super().__init__(f"Not a tool call: {content}")
        self.content = content
