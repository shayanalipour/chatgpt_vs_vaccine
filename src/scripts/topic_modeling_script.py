from common import logging_config, constants
from common.plotter import Plotter
from analysis.topic_modeling import TopicModeling

logger = logging_config.setup_logger(constants.LOGGER_NAME)


def detect_lang_ig():
    topic = "chatgpt"
    platform = "twitter"
    logger.info(f"Processing {topic} topic")
    topic_modeling = TopicModeling(platform=platform, topic=topic, detect_lang=False)
    topic_modeling.run()
    logger.info("-" * 100)


if __name__ == "__main__":
    detect_lang_ig()
