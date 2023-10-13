import os
import pandas as pd
import logging
from common import constants
from .base_data_loader import BaseDataLoader

logger = logging.getLogger(constants.LOGGER_NAME)


class FacebookDataLoader(BaseDataLoader):
    """
    This class loads Facebook data for the ChatGPT topic.
    """

    def remove_facebook_spam(self):
        """Detect and remove spam Facebook posts from the data."""
        if self.platform != "facebook":
            return

        logger.info("Starting spam detection for Facebook posts.")
        self.data = self.data.copy()

        # Initialize a new spam column with 0
        self.data["spam"] = 0

        # Convert text column to string
        self.data["text"] = self.data["text"].astype(str)

        # Define spam patterns
        spam_patterns = [
            "Video Funny Amazing #fyp #viral",
            "#reeel #cr7# #chatgpt",
            "#reels #chatgpt",
            "https://www.facebook.com/100076267686928/posts/202421482310107",
        ]

        # If a row's text contains any of the spam patterns, set spam = 1
        for pattern in spam_patterns:
            self.data.loc[
                self.data["text"].str.contains(pattern, case=False, na=False), "spam"
            ] = 1

        spam_counts = self.data["spam"].value_counts()
        logger.info(
            f"Detected {spam_counts.get(1, 0)} spam posts and {spam_counts.get(0, 0)} non-spam posts."
        )

        # Filter out spam posts
        self.data = self.data[self.data["spam"] == 0]
        self.data.drop(columns=["spam"], inplace=True)

        logger.info(f"Spam posts removed. Remaining posts count: {len(self.data)}.")

    def process_data(self):
        # load data
        self.load_data()

        # transform data
        self.transform_data()

        # remove spam
        self.remove_facebook_spam()

        # save data to database
        self.save_to_db()
