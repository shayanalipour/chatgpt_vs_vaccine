from common import logging_config, constants
from common.plotter import Plotter
from analysis.logistic_model import LogisticModel

logger = logging_config.setup_logger(constants.LOGGER_NAME)


def model_user_growth():
    logistic_model = LogisticModel(topic="all", platform="all")
    logistic_model.run()


if __name__ == "__main__":
    model_user_growth()
