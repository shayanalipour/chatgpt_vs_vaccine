from common import directories, logging_config, constants
from data_loader.reddit_dump_loader import RedditDumpLoader

logger = logging_config.setup_logger(constants.LOGGER_NAME)


def load_reddit_data():
    platforms = ["reddit"]
    for platform in platforms:
        logger.info(f"Loading data for {platform}")
        file_path = directories.REDDIT_DUMP_DIR
        loader = RedditDumpLoader(file_path=file_path, platform=platform, topic="covid")
        loader.transform_data()
        logger.info("-" * 100)


if __name__ == "__main__":
    load_reddit_data()
