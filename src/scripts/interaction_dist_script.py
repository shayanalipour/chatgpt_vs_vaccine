from common import logging_config, constants
from analysis.interaction_dist import InteractionDist

logger = logging_config.setup_logger(constants.LOGGER_NAME)


def count_daily_posts():
    for topic in constants.TOPIC_TABLES:
        logger.info(f"Processing {topic['schema_name']} schema")
        for table in topic["table_name"]:
            if table == "news":
                continue
            logger.info(f"Processing {topic['schema_name']}.{table} table")
            daily_posts = InteractionDist(topic["schema_name"], table)
            daily_posts.process_interaction_dist()
            logger.info("-" * 100)
        logger.info("=" * 100)


if __name__ == "__main__":
    count_daily_posts()
