import logging

import pandas as pd


class Threshold:
    def __init__(self, threshold: float = 0.3):
        self.threshold = threshold

    @classmethod
    def from_config(cls, model, distance, strategy, algorithm, conf_path: str, default_value: float = .3):
        logger = logging.getLogger(__name__)
        if not conf_path:
            logger.info(f"Using {strategy} strategy with threshold: {default_value}")
            return cls(threshold=default_value)

        config = pd.read_csv(conf_path)
        try:
            threshold = config[
                (config["model"] == model) &
                (config["distance"] == distance.__name__) &
                (config["strategy"] == strategy) &
                (config["algorithm"] == algorithm.__name__)].values[0]

        except IndexError:
            logger.info(f"Using {strategy} strategy with threshold: {default_value}")
            return cls(threshold=default_value)

        logger.info(f"Using {strategy} strategy with threshold: {threshold}")
        return cls(threshold=threshold)
