import logging
import numpy as np
import pandas as pd
from psycopg2 import sql
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error
from scipy.integrate import quad
import matplotlib.pyplot as plt

from common import constants, db_config, directories
from common.plotter import Plotter

logger = logging.getLogger(constants.LOGGER_NAME)


class LogisticModel:
    def __init__(self, platform: None, topic: None):
        self.platform = platform
        self.topic = topic
        self.db = db_config.Database()

    def run(self):
        if self.topic == "covid":
            self.run_covid()
        elif self.topic == "chatgpt":
            self.run_chatgpt()
        elif self.topic == "all":
            self.run_all()

    def run_all(self):
        data = self.load_all_data()
        data = pd.DataFrame(
            data,
            columns=["date", "cumulative_unique_users", "platform", "topic"],
        )

        fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(30, 20))
        fig.subplots_adjust(wspace=0.3, hspace=0.3)

        platforms = ["facebook", "instagram", "news", "reddit", "twitter", "youtube"]
        for idx, platform in enumerate(platforms):
            ax = axs[idx // 3][idx % 3]

            subset = data[data["platform"] == platform]
            covid_subset = subset[subset["topic"] == "covid"]
            chatgpt_subset = subset[subset["topic"] == "chatgpt"]

            covid_subset = self.process_data(covid_subset)
            chatgpt_subset = self.process_data(chatgpt_subset)

            # Fit the sigmoid function
            alpha, beta, popt, residuals, rmse = self.fit_sigmoid(
                covid_subset, platform
            )
            (
                gpt_alpha,
                gpt_beta,
                chatgpt_popt,
                gpt_residuals,
                gpt_rmse,
            ) = self.fit_sigmoid(chatgpt_subset, platform)
            logger.info(
                f"COVID: alpha: {alpha}, beta: {beta}, residuals: {residuals}, rmse: {rmse}"
            )
            logger.info(
                f"GPT: alpha: {gpt_alpha}, beta: {gpt_beta}, residuals: {gpt_residuals}, rmse: {gpt_rmse}"
            )

            # Calculate the AUC
            auc = self.AUC_sigmoid(covid_subset, popt)
            gpt_auc = self.AUC_sigmoid(chatgpt_subset, chatgpt_popt)
            logger.info(f"COVID: auc: {auc}, GPT: auc: {gpt_auc}")

            # Plot the sigmoid function
            plotter = Plotter()
            ax = plotter.logistic_model_plot(
                covid_subset,
                x_col="day",
                y_col="cumulative_unique_users",
                platform=platform,
                topic="covid",
                ax=ax,
                sigmoid_fit=self.sigmoid(covid_subset["day"], *popt),
                chatgpt_data=chatgpt_subset,
                chatgpt_sigmoid_fit=self.sigmoid(chatgpt_subset["day"], *chatgpt_popt),
            )
            logger.info(f"Finished plotting {platform}")
            logger.info("-" * 100)

        fig.tight_layout()
        fig.savefig(f"{directories.PLOT_DIR}/all_logistic_model.pdf", format="pdf")

    def run_covid(self):
        # Load covid data
        covid_data = self.load_data(self.platform, self.topic)
        covid_df = self.process_data(covid_data)

        # Load chatgpt data
        chatgpt_data = self.load_data(self.platform, "chatgpt")
        chatgpt_df = self.process_data(chatgpt_data)

        # Fit the sigmoid function
        alpha, beta, popt, residuals, rmse = self.fit_sigmoid(covid_df, self.platform)
        _, _, chatgpt_popt, _, _ = self.fit_sigmoid(chatgpt_df, self.platform)
        if popt is None:
            logger.error(f"Error fitting sigmoid for {self.platform}")
            raise ValueError(f"Error fitting sigmoid for {self.platform}")

        logger.info(
            f"alpha: {alpha}, beta: {beta}, residuals: {residuals}, rmse: {rmse}"
        )

        # Calculate the AUC
        auc = self.AUC_sigmoid(covid_df, popt)
        logger.info(f"auc: {auc}")

        # Plot the sigmoid function
        plotter = Plotter()
        plotter.logistic_model_plot(
            covid_df,
            x_col="day",
            y_col="cumulative_unique_users",
            platform=self.platform,
            topic=self.topic,
            sigmoid_fit=self.sigmoid(covid_df["day"], *popt),
            chatgpt_data=chatgpt_df,
            chatgpt_sigmoid_fit=self.sigmoid(chatgpt_df["day"], *chatgpt_popt),
        )

    def run_chatgpt(self):
        # Load the data
        data = self.load_data(self.platform, self.topic)
        df = self.process_data(data)

        # Fit the sigmoid function
        alpha, beta, popt, residuals, rmse = self.fit_sigmoid(df, self.platform)
        if popt is None:
            logger.error(f"Error fitting sigmoid for {self.platform}")
            raise ValueError(f"Error fitting sigmoid for {self.platform}")

        logger.info(
            f"alpha: {alpha}, beta: {beta}, residuals: {residuals}, rmse: {rmse}"
        )

        # Calculate the AUC
        auc = self.AUC_sigmoid(df, popt)
        logger.info(f"auc: {auc}")

        # Plot the sigmoid function
        plotter = Plotter()
        plotter.logistic_model_plot(
            df,
            x_col="day",
            y_col="cumulative_unique_users",
            platform=self.platform,
            topic=self.topic,
            sigmoid_fit=self.sigmoid(df["day"], *popt),
        )

    def load_all_data(self):
        query = sql.SQL(
            """
            SELECT
                *
            FROM
                processed.daily_user_counts
            ORDER BY
                date;
            """
        )

        self.db.cursor.execute(query)
        result = self.db.cursor.fetchall()

        return result

    def load_data(self, platform=None, topic=None):
        query = sql.SQL(
            """
            SELECT
                *
            FROM
                processed.daily_user_counts
            WHERE
                platform = %s AND topic = %s
            ORDER BY
                date;
            """
        )

        self.db.cursor.execute(query, (platform, topic))
        result = self.db.cursor.fetchall()

        return result

    def process_data(self, df):
        # Instead of dates (i.e. 2020-01-01), we want to use the number of days since the first date (i.e. 0, 1, 2, ...)
        df["date"] = pd.to_datetime(df["date"])
        df["day"] = (df["date"] - df["date"].min()).dt.days
        df["day"] = df["day"].astype(int)

        # Normalize the cumulative unique users
        # We use MinMax scaling to normalize the values between 0 and 1
        df["cumulative_unique_users"] = (
            df["cumulative_unique_users"] - df["cumulative_unique_users"].min()
        ) / (df["cumulative_unique_users"].max() - df["cumulative_unique_users"].min())

        return df

    def sigmoid(self, t, alpha, beta):
        return 1 / (1 + np.exp(-alpha * (t - beta)))

    def fit_sigmoid(self, platform_df, platform_name):
        initial_values = {
            "facebook": [0.1, 70],
            "instagram": [0.07, 60],
            "news": [0.07, 60],
            "default": None,
        }

        try:
            popt, _ = curve_fit(
                self.sigmoid,
                platform_df["day"],
                platform_df["cumulative_unique_users"],
                maxfev=10000,
                p0=initial_values.get(platform_name, initial_values["default"]),
            )

            residuals = platform_df["cumulative_unique_users"] - self.sigmoid(
                platform_df["day"], *popt
            )
            rmse = np.sqrt(
                mean_squared_error(
                    platform_df["cumulative_unique_users"],
                    self.sigmoid(platform_df["day"], *popt),
                )
            )

            return (
                round(popt[0], 3),
                round(popt[1], 3),
                popt,
                round(np.sum(residuals**2), 3),
                round(rmse, 3),
            )
        except Exception as e:
            logger.error(f"Error fitting sigmoid for {platform_name}: {e}")

    def AUC_sigmoid(self, platform_df, popt):
        T = platform_df["day"].max()
        area, _ = quad(self.sigmoid, 0, T, args=(popt[0], popt[1]))
        return round(area / T, 3)
