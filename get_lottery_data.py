from typing import Optional, List, Dict, Any, NewType
from datetime import datetime, timedelta
import requests
from threading import Lock
import concurrent.futures

# Constants for date range
MAX_START_DATE = "2002-07-04"
MAX_END_DATE = datetime.now().strftime("%Y-%m-%d")

# Type alias for a lottery draw
LotteryDraw = NewType("LotteryDraw", Dict[str, Any])


def get_ball_color(number: int) -> str:
    """
    Returns the color for a given ball number based on Mark Six rules.

    Args:
        number (int): Ball number (1-49)

    Returns:
        str: Color ('red', 'blue', or 'green')
    """
    color_index = ((number - 1) + ((number - 1) // 10)) % 6 // 2
    if color_index == 0:
        return "red"
    elif color_index == 1:
        return "blue"
    else:
        return "green"


ball_number_color_mapping: Dict[int, str] = {i: get_ball_color(i) for i in range(1, 50)}


def _fetch_lottery_data(variables: dict) -> List[LotteryDraw]:
    """
    Fetch lottery data from the HKJC API.

    Args:
        variables (dict): Query parameters for the API request. Can include:
            - lastNDraw (int): Number of most recent draws to fetch.
            - startDate (str): Start date in 'YYYYMMDD' format.
            - endDate (str): End date in 'YYYYMMDD' format.

    Returns:
        List[LotteryDraw]: List of lottery draws, each as a dictionary.
    """
    url = "https://info.cld.hkjc.com/graphql/base/"
    data = {
        "operationName": "marksixResult",
        "variables": variables,
        "query": """fragment lotteryDrawsFragment on LotteryDraw {
            id
            year
            no
            openDate
            closeDate
            drawDate
            status
            snowballCode
            snowballName_en
            snowballName_ch
            lotteryPool {
                sell
                status
                totalInvestment
                jackpot
                unitBet
                estimatedPrize
                derivedFirstPrizeDiv
                lotteryPrizes {
                type
                winningUnit
                dividend
                }
            }
            drawResult {
                drawnNo
                xDrawnNo
            }
        }

        query marksixResult($lastNDraw: Int, $startDate: String, $endDate: String, $drawType: LotteryDrawType) {
            lotteryDraws(
                lastNDraw: $lastNDraw
                startDate: $startDate
                endDate: $endDate
                drawType: $drawType
            ) {
                ...lotteryDrawsFragment
            }
        }""",
    }
    response = requests.post(url=url, json=data)
    if response.status_code != 200:
        raise Exception(
            f"Error fetching data: {response.status_code} - {response.text}"
        )
    return response.json()["data"]["lotteryDraws"]


def get_lottery_data(
    last_n_draw: Optional[int] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> List[LotteryDraw]:
    """
    Retrieve Mark Six lottery draw data from the HKJC API.

    Args:
        last_n_draw (Optional[int]): Number of most recent draws to fetch.
        start_date (Optional[str]): Start date in 'YYYY-MM-DD' format.
        end_date (Optional[str]): End date in 'YYYY-MM-DD' format.

    Raises:
        ValueError: If neither last_n_draw nor both start_date and end_date are provided,
            or if the date range is out of bounds.

    Returns:
        List[LotteryDraw]: List of lottery draws, each as a dictionary.
    """
    if last_n_draw is not None:
        # Fetch the last N draws
        variables = {"lastNDraw": last_n_draw}
        print(f"Fetching the last {last_n_draw} draws")
        data: List[LotteryDraw] = _fetch_lottery_data(variables)
    elif start_date is not None and end_date is not None:
        # Fetch by date range
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")

        # Validate date range
        if start_dt < datetime.strptime(
            MAX_START_DATE, "%Y-%m-%d"
        ) or end_dt > datetime.strptime(MAX_END_DATE, "%Y-%m-%d"):
            raise ValueError(
                f"Date range must be between {MAX_START_DATE} and {MAX_END_DATE}"
            )

        # If range > 3 months, split into 3-month chunks for API
        if (end_dt - start_dt).days > 90:
            print(
                f"Date range is larger than 3 months, splitting the request from {start_dt} to {end_dt}"
            )
            data: List[LotteryDraw] = []
            data_lock = Lock()
            date_ranges = []
            current_start = start_dt
            while current_start < end_dt:
                current_end = min(current_start + timedelta(days=90), end_dt)
                date_ranges.append((current_start, current_end))
                current_start = current_end + timedelta(days=1)

            def fetch_data_for_range(date_range):
                chunk_start, chunk_end = date_range
                range_variables = {
                    "startDate": chunk_start.strftime("%Y%m%d"),
                    "endDate": chunk_end.strftime("%Y%m%d"),
                }
                print(
                    f"Fetching data from {range_variables['startDate']} to {range_variables['endDate']}"
                )
                return _fetch_lottery_data(range_variables)

            # Parallel requests for each chunk
            with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
                futures = [
                    executor.submit(fetch_data_for_range, date_range)
                    for date_range in date_ranges
                ]
                for future in concurrent.futures.as_completed(futures):
                    try:
                        result = future.result()
                        with data_lock:
                            data.extend(result)
                    except Exception as e:
                        print(f"Error fetching data: {e}")
        else:
            variables = {
                "startDate": start_dt.strftime("%Y%m%d"),
                "endDate": end_dt.strftime("%Y%m%d"),
            }
            print(
                f"Fetching data from {variables['startDate']} to {variables['endDate']}"
            )
            data: List[LotteryDraw] = _fetch_lottery_data(variables)
    else:
        raise ValueError("Either provide last_n_draw or both start_date and end_date")

    # Sort by drawDate in descending order
    data.sort(
        key=lambda x: datetime.strptime(x["drawDate"], "%Y-%m-%d+08:00"), reverse=True
    )
    return data


if __name__ == "__main__":
    # Example usage
    # draws = get_lottery_data(last_n_draw=10)
    draws = get_lottery_data(start_date="2010-11-08", end_date="2025-05-29")
    print(f"Fetched {len(draws)} draws.")
