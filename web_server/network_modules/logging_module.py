"""로그를 전용 스레드에서 출력하는 모듈."""

import logging
import queue
from logging.handlers import QueueHandler, QueueListener


def configure_logging(app):
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(threadName)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    console_handler.terminator = "\n\n"

    log_queue = queue.Queue()
    queue_handler = QueueHandler(log_queue)

    logger = logging.getLogger("web_server.refactored")
    for target_logger in (logger, app.logger, logging.getLogger("werkzeug")):
        target_logger.handlers.clear()
        target_logger.addHandler(queue_handler)
        target_logger.setLevel(logging.INFO)
        target_logger.propagate = False

    listener = QueueListener(
        log_queue,
        console_handler,
        respect_handler_level=True,
    )
    listener.start()
    return logger, listener
