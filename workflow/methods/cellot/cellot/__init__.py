import logging

# Setup package logger
logging.basicConfig(
    level='INFO',
    format='%(filename)s | %(message)s',
)

log = logging.getLogger(__name__)
