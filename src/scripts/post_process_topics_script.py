from common import logging_config, constants
from common.plotter import Plotter
from analysis.post_process_topics import PostProcessTopics

logger = logging_config.setup_logger(constants.LOGGER_NAME)


def post_process_topics():
    platforms = [
        "instagram",
        "youtube",
        "reddit",
        "facebook",
        "twitter_batch1",
        "twitter_batch2",
    ]
    topic = "chatgpt"
    for platform in platforms:
        if "twitter" in platform:
            logger.info(f"Processing {platform} platform")
            post_process_topics = PostProcessTopics(platform=platform, topic=topic)
            post_process_topics.update_count()
            post_process_topics.add_topic_name()
            logger.info("-" * 100)


def summary_table():
    post_process_topics = PostProcessTopics(platform="all", topic="chatgpt")
    # post_process_topics.create_summary_table()
    post_process_topics.plot_topic_counts()


if __name__ == "__main__":
    summary_table()
