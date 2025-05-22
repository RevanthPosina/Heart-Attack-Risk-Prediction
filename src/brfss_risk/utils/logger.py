# utils/logger.py

import logging
import sys
import json
import datetime

def get_logger(name="brfss"):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
        stream=sys.stdout,
    )
    return logging.getLogger(name)

logger = get_logger()

def log_event(phase, action, summary_dict):
    record = {
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "phase": phase,
        "action": action,
        **summary_dict
    }
    logger.info(json.dumps(record))
