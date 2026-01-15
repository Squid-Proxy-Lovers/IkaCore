import logging


def configure(level: int | str = "INFO", force: bool = False) -> None:
    current_level = getattr(configure, "_current_level", None)
    already_configured = getattr(configure, "_configured", False)

    if already_configured and not force and current_level == level:
        return

    configure._current_level = level

    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    logging.basicConfig(level=level, format="%(asctime)s %(name)s %(levelname)s: %(message)s")
    configure._configured = True

    logger = logging.getLogger(__name__)
    logger.info(f"Logging configured with level={level}")