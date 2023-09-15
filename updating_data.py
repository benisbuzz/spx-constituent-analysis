import pandas as pd
import nasdaqdatalink as nq
import os
import structlog
from dotenv import load_dotenv
from pathlib import Path
from tqdm import tqdm

load_dotenv()

nq.ApiConfig.api_key = os.getenv("NASDAQ_DATALINK_API_KEY")

HOME = Path(__file__).parent

logger = structlog.get_logger()


def get_existing_spx() -> pd.DataFrame:
    return pd.read_parquet(HOME / "data/sp500_time_series.parquet")


def get_constituents(start_date: pd.Timestamp) -> pd.Series:
    return nq.get_table(
        "SHARADAR/SP500",
        action=["current", "historical"],
        date={"gt": start_date.strftime("%Y-%m-%d")},
        paginate=True,
    ).set_index("date")["ticker"]


def get_new_spx(
    start_date: pd.Timestamp, unique_constituents: list[str]
) -> pd.DataFrame:
    return (
        nq.get_table(
            "SHARADAR/DAILY",
            date={"gt": start_date.strftime("%Y-%m-%d")},
            ticker=unique_constituents,
            paginate=True,
        )
        .set_index("date")
        .drop(["lastupdated", "ev", "evebit"], axis=1)
    )


def get_closest_date(
    target_date: pd.Timestamp, date_list: list[pd.Timestamp]
) -> pd.Timestamp:
    date_series = pd.Series(date_list)
    differences = abs(date_series - target_date)
    closest_date_index = differences.idxmin()
    return date_series[closest_date_index]


def get_sp_given_date(
    date: pd.Timestamp, constituents: pd.Series, time_series_raw: pd.DataFrame
) -> pd.DataFrame:
    closest_date = get_closest_date(date, list(constituents.index))
    common_tickers = list(
        set(constituents[closest_date]).intersection(time_series_raw["ticker"])
    )
    cross_section = time_series_raw.loc[date]
    sp500 = cross_section[cross_section["ticker"].isin(common_tickers)]
    index = pd.MultiIndex.from_product(
        [[date], sp500["ticker"]], names=["date", "ticker"]
    )
    sp500.index = index
    return sp500.drop("ticker", axis=1)


def get_spx() -> pd.DataFrame:
    existing_spx = get_existing_spx()
    logger.info("fetched existing spx", df_shape=existing_spx.shape)
    end_date = existing_spx.index.levels[0][-1]
    constituents = get_constituents(end_date)
    unique_constituents = list(set(constituents))
    logger.info("fetched new constituents", count=len(unique_constituents))
    new_spx = get_new_spx(end_date, unique_constituents)
    logger.info("fetched all new spx", df_shape=new_spx.shape)
    if new_spx.empty:
        logger.info("existing spx up to date", end_date=end_date)
        existing_spx.to_parquet(HOME / "data/sp500_time_series.parquet")
        return existing_spx
    dates = sorted(set(new_spx.index))
    index = 1
    for date in tqdm(dates):
        spx_new = get_sp_given_date(date, constituents, new_spx)
        logger.info(f"fetched spx {index}", date=date)
        existing_spx = pd.concat([existing_spx, spx_new])
        logger.info("added new df to existing", count=index)
        index += 1
    return existing_spx


def get_weighted_spx(unweighted_spx: pd.DataFrame) -> pd.DataFrame:
    return unweighted_spx.groupby(level=0).apply(
        lambda df: df.assign(weight=df["marketcap"] / df["marketcap"].sum())
        .sort_values(by="marketcap", ascending=False)
        .reset_index(level=0, drop=True)
    )
