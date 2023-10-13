import os
import pandas as pd
import logging
import tldextract


from common import constants
from .base_data_loader import BaseDataLoader
from .helper import convert_date

logger = logging.getLogger(constants.LOGGER_NAME)


class NewsDataLoader(BaseDataLoader):
    def transform_data(self):
        # drop nan values for id and date
        self.data = self.data.dropna(subset=["id", "date"]).reset_index(drop=True)

        # Split the tone strings to get the average tone and polarity score
        tones_list = self.data["tone"].str.split(",")
        self.data["avr_tone"] = tones_list.str[0].astype(float).round(2)
        self.data["polarity"] = tones_list.str[3].astype(float).round(2)
        self.data = self.data.drop(columns=["tone"])

        # convert date column
        self.data["date"] = pd.to_datetime(
            self.data["date"], errors="coerce", format="%Y%m%d%H%M%S"
        ).dt.date
        self.data = self.data.dropna(subset=["date"])
        self.data = self.data.sort_values(by=["date"])

        # select time range
        start_date = convert_date(constants.DATE_RANGE[self.topic]["start"])
        end_date = convert_date(constants.DATE_RANGE[self.topic]["end"])
        self.data = self.data[
            (self.data["date"] >= start_date) & (self.data["date"] <= end_date)
        ]
        logger.info(f"Min date: {self.data['date'].min()}")
        logger.info(f"Max date: {self.data['date'].max()}")
        logger.info("Processed date column")

        # Add domain column
        self.data["domain"] = self.data["id"].apply(
            lambda x: tldextract.extract(x).domain + "." + tldextract.extract(x).suffix
        )

        # remove duplicates
        self.data = self.data.drop_duplicates(subset=["id"]).reset_index(drop=True)
