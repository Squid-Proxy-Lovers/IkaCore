import logging

from ..runtime.workflow import Workflow

_LOG = logging.getLogger(__name__)


class DummyWorkflow(Workflow):
    @classmethod
    def add_arguments(cls, group):
        group.add_argument(
            "--command",
            help="Command to run",
            default="ls -la /opt/resources",
            type=str,
        )

    def run(self, env):
        self.logger.info("Running dummy command: %s", self.kwargs.get("command"))
        result = env.exec(["sh", "-lc", self.kwargs.get("command")])
        return result.stdout


