import os
import pandas as pd
import logging
from common import constants
from .base_data_loader import BaseDataLoader

logger = logging.getLogger(constants.LOGGER_NAME)


class CovidFacebookDataLoader(BaseDataLoader):
    def __init__(self, file_path: str, topic: str, platform: str):
        super().__init__(file_path, topic, platform)
        self.file_names = [f for f in os.listdir(self.file_path) if f.endswith(".csv")]
        logger.info(f"Found {len(self.file_names)} files in {self.file_path}")

    def load_data(self):
        """
        Loads Facebook data from multiple files.
        """
        dataframes = []
        for file in self.file_names:
            full_path = f"{self.file_path}/{file}"
            try:
                df = self.load_single_fb_file(full_path, file)
                dataframes.append(df)
            except Exception as e:
                logger.error(f"Error loading {self.platform} data: {e}")
                continue
        self.data = pd.concat(dataframes, ignore_index=True)
        self.data = self.data.drop_duplicates(subset=["id"]).reset_index(drop=True)
        logger.info(f"Total records: {len(self.data)}")
        logger.info(f"Successfully loaded {self.platform} data from {self.file_path}")

    def load_single_fb_file(self, file_path: str, file_name: str):
        """
        Loads a single Facebook file.
        """
        if file_name == "FB_post_pages_IT_vaccines.csv":
            col_map = {
                "Facebook Id": "author_id",
                "URL": "id",
                "Post Created": "date",
                "Likes": "interaction_1",
                "Comments": "interaction_2",
            }
        else:
            col_map = {
                "url": "id",
                "account_id": "author_id",
                "created_at": "date",
                "likes": "interaction_1",
                "comments": "interaction_2",
            }

        df = pd.read_csv(file_path, usecols=col_map.keys(), low_memory=False)
        df.rename(columns=col_map, inplace=True)
        return df
