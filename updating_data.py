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


def get_parquet(file_name: str) -> pd.DataFrame:
    return pd.read_parquet(HOME / f"data/{file_name}")


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
    date: pd.Timestamp, constituents: list, time_series_raw: pd.DataFrame
) -> pd.DataFrame:
    assert (
        date in time_series_raw.index
    ), "not valid date. make sure date is in index of time series dataframe"
    common_tickers = list(set(constituents).intersection(time_series_raw["ticker"]))
    cross_section = time_series_raw.loc[date]
    sp500 = cross_section[cross_section["ticker"].isin(common_tickers)]
    index = pd.MultiIndex.from_product(
        [[date], sp500["ticker"]], names=["date", "ticker"]
    )
    sp500.index = index
    return sp500.drop("ticker", axis=1)


def get_spx() -> pd.DataFrame:
    existing_spx = get_parquet("sp500_time_series.parquet")
    end_date = existing_spx.index.get_level_values(0)[-1]
    logger.info(
        "fetched existing spx",
        start_date=existing_spx.index.get_level_values(0)[0],
        end_date=end_date,
        df_shape=existing_spx.shape,
    )
    all_constituents = get_constituents(end_date)
    unique_constituents = list(set(all_constituents))
    logger.info("fetched new constituents", count=len(unique_constituents))
    new_spx = get_new_spx(end_date, unique_constituents)
    if new_spx.empty:
        logger.info("existing spx up to date", end_date=end_date)
        return existing_spx
    logger.info(
        "fetched all new spx",
        start_date=new_spx.index.get_level_values(0)[0],
        end_date=new_spx.index.get_level_values(0)[-1],
        df_shape=new_spx.shape,
    )
    index = 1
    for date in sorted(list(set(new_spx.index))):
        spx_new = get_sp_given_date(date, all_constituents, new_spx)
        logger.info(f"fetched spx {index}", date=date)
        existing_spx = pd.concat([existing_spx, spx_new])
        logger.info("added new df to existing", count=index)
        index += 1
    return existing_spx


def get_weighted_spx(unweighted_spx: pd.DataFrame) -> pd.DataFrame:
    existing_weighted_spx = get_parquet("sp500_time_series_weighted.parquet")
    logger.info(
        "fetched existing weighted spx",
        start_date=existing_weighted_spx.index.get_level_values(0)[0],
        end_date=existing_weighted_spx.index.get_level_values(0)[-1],
        df_shape=existing_weighted_spx.shape,
    )
    new_spx = unweighted_spx[
        unweighted_spx.index.get_level_values("date")
        > existing_weighted_spx.index.levels[0][-1]
    ]
    if new_spx.empty:
        logger.info(
            "existing sdx up to date",
            end_date=existing_weighted_spx.index.levels[0][-1],
        )
        return existing_weighted_spx
    logger.info(
        "weighting new spx",
        start_date=new_spx.index.get_level_values(0)[0],
        end_date=new_spx.index.get_level_values(0)[-1],
        df_shape=new_spx.shape,
    )
    return pd.concat(
        [
            existing_weighted_spx,
            new_spx.groupby(level=0).apply(
                lambda df: df.assign(weight=df["marketcap"] / df["marketcap"].sum())
                .sort_values(by="marketcap", ascending=False)
                .reset_index(level=0, drop=True)
            ),
        ]
    )


def get_ndx(spx: pd.DataFrame) -> pd.DataFrame:
    existing_ndx = get_parquet("nasdaq_time_series.parquet")
    logger.info(
        "fetched existing ndx",
        start_date=existing_ndx.index.get_level_values(0)[0],
        end_date=existing_ndx.index.get_level_values(0)[-1],
        df_shape=existing_ndx.shape,
    )
    new_spx = spx[spx.index.get_level_values("date") > existing_ndx.index.levels[0][-1]]
    if new_spx.empty:
        logger.info(
            "existing ndx up to date", end_date=existing_ndx.index.get_level_values(0)[-1]
        )
        return existing_ndx
    tickers = new_spx.index.get_level_values(1).unique().to_list()
    logger.info("fetched spx tickers", count=len(tickers))
    tickers_metadata = nq.get_table(
        "SHARADAR/TICKERS", table="SF1", ticker=tickers, paginate=True
    )
    logger.info("fetched ticker metadata", df_shape=tickers_metadata.shape)
    potential_tickers = list(
        tickers_metadata[tickers_metadata["exchange"] == "NASDAQ"]["ticker"]
    )
    logger.info("fetched potential tickers", count=len(potential_tickers))
    mask = new_spx.index.get_level_values("ticker").isin(potential_tickers)
    return pd.concat(
        [
            existing_ndx,
            new_spx.loc[mask]
            .groupby(level="date")
            .apply(lambda df: df.sort_values(by="marketcap", ascending=False).head(100))
            .reset_index(level=0, drop=True),
        ]
    )


def get_weighted_ndx(ndx: pd.DataFrame) -> pd.DataFrame:
    existing_weighted_ndx = get_parquet("nasdaq_time_series_weighted.parquet")
    logger.info(
        "fetched existing weighted ndx",
        start_date=existing_weighted_ndx.index.get_level_values(0)[0],
        end_date=existing_weighted_ndx.index.get_level_values(0)[-1],
        df_shape=existing_weighted_ndx.shape,
    )
    new_ndx = ndx[
        ndx.index.get_level_values(0)
        > existing_weighted_ndx.index.get_level_values(0)[-1]
    ]
    if new_ndx.empty:
        logger.info(
            "existing ndx up to date", end_date=existing_weighted_ndx.index.get_level_values(0)[-1]
        )
        return existing_weighted_ndx
    logger.info(
        "fetched new ndx",
        start_date=new_ndx.index.get_level_values(0)[-1],
        end_date=new_ndx.index.get_level_values(0)[-1],
        df_shape=new_ndx.shape,
    )
    return pd.concat(
        [
            existing_weighted_ndx,
            new_ndx.groupby(level=0).apply(
                lambda df: df.assign(weight=df["marketcap"] / df["marketcap"].sum())
            ).reset_index(level=1, drop=True)
        ]
    )
