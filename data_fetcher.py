import click
import os
import pandas as pd
import random
import time
import urllib2

from BeautifulSoup import BeautifulSoup
from datetime import datetime

# Directory to save the data files
DATA_DIR = "data"

# Range for random sleep times between requests to avoid being blocked
RANDOM_SLEEP_TIMES = (1, 5)

# URL to fetch the list of S&P 500 companies
SP500_LIST_URL = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents-financials.csv"

# Path to save the downloaded list of S&P 500 companies
SP500_LIST_PATH = os.path.join(DATA_DIR, "constituents-financials.csv")


def _download_sp500_list():
    """
    Download the list of S&P 500 companies if not already downloaded.
    """
    if os.path.exists(SP500_LIST_PATH):
        return  # If file already exists, no need to download

    # Fetch the list from the URL
    f = urllib2.urlopen(SP500_LIST_URL)
    print "Downloading ...", SP500_LIST_URL
    with open(SP500_LIST_PATH, 'w') as fin:
        print >> fin, f.read()  # Save the content to a file
    return


def _load_symbols():
    """
    Load the stock symbols from the downloaded S&P 500 companies list.
    """
    _download_sp500_list()  # Ensure the list is downloaded
    df_sp500 = pd.read_csv(SP500_LIST_PATH)  # Read the CSV file
    df_sp500.sort('Market Cap', ascending=False, inplace=True)  # Sort by Market Cap in descending order
    stock_symbols = df_sp500['Symbol'].unique().tolist()  # Get unique stock symbols
    print "Loaded %d stock symbols" % len(stock_symbols)  # Print the number of symbols loaded
    return stock_symbols


def fetch_prices(symbol, out_name):
    """
    Fetch daily stock prices for the given stock symbol since 1980-01-01.

    Args:
        symbol (str): a stock abbreviation symbol, like "GOOG" or "AAPL".
        out_name (str): output file name to save the fetched data.

    Returns:
        bool: Whether the fetch succeeded.
    """
    # Format today's date to match Google's finance history API
    now_datetime = datetime.now().strftime("%b+%d,+%Y")

    # URL for fetching historical stock prices
    BASE_URL = "https://finance.google.com/finance/historical?output=csv&q={0}&startdate=Jan+1%2C+1980&enddate={1}"
    symbol_url = BASE_URL.format(
        urllib2.quote(symbol),
        urllib2.quote(now_datetime, '+')
    )
    print "Fetching {} ...".format(symbol)
    print symbol_url

    try:
        # Fetch the data from the URL
        f = urllib2.urlopen(symbol_url)
        with open(out_name, 'w') as fin:
            print >> fin, f.read()  # Save the fetched data to a file
    except urllib2.HTTPError:
        print "Failed when fetching {}".format(symbol)
        return False  # Return False if there's an HTTP error

    data = pd.read_csv(out_name)  # Read the saved CSV file
    if data.empty:
        print "Remove {} because the data set is empty.".format(out_name)
        os.remove(out_name)  # Remove the file if it's empty
    else:
        dates = data.iloc[:,0].tolist()  # Get the dates from the data
        print "# Fetched rows: %d [%s to %s]" % (data.shape[0], dates[-1], dates[0])  # Print the number of fetched rows

    # Random sleep to avoid being blocked
    sleep_time = random.randint(*RANDOM_SLEEP_TIMES)
    print "Sleeping ... %ds" % sleep_time
    time.sleep(sleep_time)
    return True


@click.command(help="Fetch stock prices data")
@click.option('--continued', is_flag=True)
def main(continued):
    """
    Main function to fetch stock prices data.

    Args:
        continued (bool): Whether to continue fetching data if previously fetched data exists.
    """
    random.seed(time.time())  # Seed the random number generator
    num_failure = 0  # Counter for failures

    # This is S&P 500 index
    #fetch_prices('INDEXSP%3A.INX')

    symbols = _load_symbols()  # Load the stock symbols
    for idx, sym in enumerate(symbols):
        out_name = os.path.join(DATA_DIR, sym + ".csv")  # Set the output file name
        if continued and os.path.exists(out_name):
            print "Fetched", sym  # Skip if continued flag is set and file already exists
            continue

        succeeded = fetch_prices(sym, out_name)  # Fetch prices for the symbol
        num_failure += int(!succeeded)  # Increment failure count if fetch failed

        if idx % 10 == 0:  # Print the number of failures every 10 symbols
            print "# Failures so far [%d/%d]: %d" % (idx + 1, len(symbols), num_failure)


if __name__ == "__main__":
    main()  # Run the main function if this script is executed
