from __future__ import annotations

from typing import Dict

from IkaCore.cli_output import get_cli_output
from IkaModel.base import BareBoneModel
from IkaModel.chat_interface import summarise_message_history


class AgentExecutionMixin:

    def _build_final_output(self, final_message: str, barebone_model: BareBoneModel) -> Dict[str, str]:
        summary = ""
        if self.enable_summarization:
            summary = summarise_message_history(barebone_model, self.message_history) or self.message_history.get("summary", {}).get("message", "")
            if summary:
                if self.logger:
                    self.logger.log_summary(summary)
                # Also emit via CLI output with dedicated summarization color and no truncation
                cli = get_cli_output()
                current_hierarchy = getattr(self, "_parent_hierarchy", []) + [self.name]
                # Use final step number for summary box
                step = cli.get_step(self.name) or 0
                cli.summarization(self.name, summary, current_hierarchy, step=step)
        
        if not final_message or final_message.strip() == "":
            final_message = summary
        
        return {"final_message": final_message, "summary": summary}

