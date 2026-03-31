import logging
import warnings


MMCV_WARNING_PATTERN = r'.*MMCV will release v2\.0\.0.*'
TUTEL_WARNING_MESSAGE = 'Cannot import JIT optimized kernels. CUDA extension will be disabled.'


class _MessageFilter(logging.Filter):
    def __init__(self, blocked_message):
        super().__init__()
        self.blocked_message = blocked_message

    def filter(self, record):
        return record.getMessage() != self.blocked_message


def configure_warning_filters():
    warnings.filterwarnings('ignore', message=MMCV_WARNING_PATTERN, category=UserWarning)

    root_logger = logging.getLogger()
    if not any(
        isinstance(existing_filter, _MessageFilter)
        and existing_filter.blocked_message == TUTEL_WARNING_MESSAGE
        for existing_filter in root_logger.filters
    ):
        root_logger.addFilter(_MessageFilter(TUTEL_WARNING_MESSAGE))

configure_warning_filters()
