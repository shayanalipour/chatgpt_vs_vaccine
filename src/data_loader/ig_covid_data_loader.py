import os
import pandas as pd
import logging
from common import constants
from .base_data_loader import BaseDataLoader
from .helper import (
    convert_date,
    merge_text,
    merge_interactions,
)

logger = logging.getLogger(constants.LOGGER_NAME)


class InstagramDataLoader(BaseDataLoader):
    """
    This class loads Instagram data for the COVID topic.
    """

    def __init__(self, file_path: str, topic: str, platform: str):
        super().__init__(file_path, topic, platform)
        self.file_names = [f for f in os.listdir(self.file_path) if f.endswith(".csv")]
        logger.info(f"Found {len(self.file_names)} files in {self.file_path}")

    def load_data(self):
        """
        Loads Instagram data from multiple files.
        """
        dataframes = []
        for file in self.file_names:
            full_path = f"{self.file_path}/{file}"
            try:
                df = self.load_single_ig_file(full_path)
                dataframes.append(df)
            except Exception as e:
                logger.error(f"Error loading {self.platform} data: {e}")
                continue
        self.data = pd.concat(dataframes, ignore_index=True)
        self.data = self.data.drop_duplicates(subset=["id"]).reset_index(drop=True)
        logger.info(f"Total records: {len(self.data)}")
        logger.info(f"Successfully loaded {self.platform} data from {self.file_path}")

    def load_single_ig_file(self, file_path: str):
        """
        Loads a single Instagram file.
        """
        column_mapping = constants.COLUMN_CONFIG[self.platform]
        df = pd.read_csv(file_path, usecols=column_mapping.keys(), low_memory=False)
        df.rename(columns=column_mapping, inplace=True)
        return df
