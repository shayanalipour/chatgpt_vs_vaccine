from common import constants, logging_config
from analysis.plot_general_counts import GeneralCountsPlotter

logger = logging_config.setup_logger(constants.LOGGER_NAME)


# Plot the daily post count and interaction distribution for each topic-platform
def plot_general_counts():
    p = GeneralCountsPlotter()
    p.plot_fig1()


if __name__ == "__main__":
    plot_general_counts()
