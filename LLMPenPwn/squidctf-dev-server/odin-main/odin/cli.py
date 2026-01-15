import argparse
import logging
import sys
from pathlib import Path

from . import logging_config
from .core.container import ContainerManager
from .runtime.environment import Environment
from .workflows import WORKFLOW_REGISTRY

_LOG = logging.getLogger(__name__)

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="odin",
        description="Odin: AI-powered vulnerability analysis for security competitions",
    )

    parser.add_argument(
        "-W", "--workflow",
        choices=list(WORKFLOW_REGISTRY.keys()),
        default="static",
        help="Analysis workflow to run (default: %(default)s)",
    )

    temp_parser = argparse.ArgumentParser(add_help=False)
    temp_parser.add_argument("-W", "--workflow")
    partial_args, _ = temp_parser.parse_known_args(argv)

    if partial_args.workflow in WORKFLOW_REGISTRY:
        wf_cls = WORKFLOW_REGISTRY[partial_args.workflow]
        wf_group = parser.add_argument_group(f"{partial_args.workflow} workflow options")
        wf_cls.add_arguments(wf_group)

    # Environment options
    parser.add_argument(
        "--deploy-service",
        action="store_true",
        help="Deploy services via docker-compose if available (auto-detect compose when not provided).",
    )
    parser.add_argument(
        "--compose-file",
        type=str,
        default=None,
        help="Path to docker-compose file (used when --deploy-service is set)",
    )
    parser.add_argument(
        "--add-sagemath",
        action="store_true",
        default=False,
        help="Add a sagemath tool to the agent. This will use the odin-sage:latest image.",
    )

    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Logging level (default: %(default)s)",
    )

    parser.add_argument(
        "code_path",
        type=str,
        help="Path to the codebase to analyze",
    )
    return parser.parse_args(argv)

def main(argv=None):
    args = parse_args(argv)
    logging_config.configure(level=args.log_level, force=True)
    _LOG.debug(f"Odin executed with log level {args.log_level}, workflow={args.workflow}")

    code_path = Path(args.code_path)
    if not code_path.is_absolute():
        code_path = Path.cwd() / code_path
    code_path = code_path.resolve()

    if not code_path.exists():
        _LOG.error("Code path does not exist: %s", code_path)
        return 1

    if args.workflow not in WORKFLOW_REGISTRY:
        _LOG.error(
            "Unknown workflow '%s'. Available workflows: %s",
            args.workflow,
            ", ".join(sorted(WORKFLOW_REGISTRY.keys()))
        )
        return 1

    wf_cls = WORKFLOW_REGISTRY[args.workflow]
    _LOG.info("Running %s on %s", wf_cls.__name__, code_path)

    kwargs = {k: v for k, v in vars(args).items() if k not in ("code_path", "workflow", "deploy_service", "compose_file", "log_level")}

    try:
        env = Environment(
            code_path,
            deploy_service=bool(getattr(args, "deploy_service", False)),
            compose_file=Path(getattr(args, "compose_file")) if getattr(args, "compose_file", None) else None,
            image=ContainerManager.DEFAULT_SAGE_IMAGE if getattr(args, "add_sagemath", False) else ContainerManager.DEFAULT_IMAGE,
        )
        with env:
            workflow = wf_cls(code_path, **kwargs)
            output = workflow.run(env)
            if output:
                _LOG.info("Analysis output:\n%s\n", output)
            else:
                _LOG.info("No output from analysis")
        return 0
    except KeyboardInterrupt:
        _LOG.warning("Analysis interrupted by user")
        return 130
    except Exception as e:
        _LOG.exception("Error during analysis: %s", e)
        return 1

if __name__ == "__main__":
    sys.exit(main())