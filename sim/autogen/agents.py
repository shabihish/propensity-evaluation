import time
from typing import AsyncGenerator, List, Sequence
from autogen_agentchat.agents import BaseChatAgent
from autogen_agentchat.base import Response
from autogen_agentchat.messages import AgentEvent, ChatMessage, TextMessage
from autogen_core import CancellationToken

class ListYieldingAgent(BaseChatAgent):
    def __init__(self, name: str, sys_messages: List[str]):
        super().__init__(name, "An agent that yields messages from a list.")
        self.sys_messages = sys_messages

    @property
    def produced_message_types(self) -> Sequence[type[ChatMessage]]:
        return (TextMessage,)

    async def on_messages(self, messages: Sequence[ChatMessage], cancellation_token: CancellationToken) -> Response:
        response: Response | None = None
        async for message in self.on_messages_stream(messages, cancellation_token):
            if isinstance(message, Response):
                response = message
        assert response is not None
        return response

    async def on_messages_stream(
            self, messages: Sequence[ChatMessage], cancellation_token: CancellationToken
    ) -> AsyncGenerator[AgentEvent | ChatMessage | Response, None]:
        inner_messages: List[AgentEvent | ChatMessage] = []
        for msg_content in self.sys_messages:
            msg = TextMessage(content=msg_content, source=self.name)
            inner_messages.append(msg)
            time.sleep(1)
            yield msg
        yield Response(chat_message=TextMessage(content="Done!", source=self.name), inner_messages=inner_messages)

    async def on_reset(self, cancellation_token: CancellationToken) -> None:
        pass

