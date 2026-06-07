# pyright: strict

from __future__ import annotations

from IkaCore.agent_chat_support import AgentChatSupportMixin
from IkaCore.agent_helper_support import AgentHelperSupportMixin
from IkaCore.agent_lifecycle import AgentLifecycleMixin


class AgentHelpersMixin(AgentHelperSupportMixin, AgentLifecycleMixin, AgentChatSupportMixin):
    pass
