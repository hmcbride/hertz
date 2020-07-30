import logging

import pandas as pd
import time
from pandas_datareader import data as wb
from pypfopt import EfficientFrontier
from pypfopt import expected_returns
from pypfopt import risk_models
from itertools import combinations


def main():
    stocks = pd.read_csv('//nasdaq3.csv', usecols=[0, 1])

    logging.basicConfig(filename='app.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s',
                        level=logging.INFO)

    tuples = dict(list(zip(*map(stocks.get, ['Symbol', 'Name']))))

    flipped = getMultiStockName(tuples)

    single_ticker = dict(filter(lambda elem: len(elem[1]) == 1, flipped.items()))
    single_ticker_clean = sorted(set([k for i, j in single_ticker.items() for k in j if k]))

    multi_ticker = dict(filter(lambda elem: len(elem[1]) > 1, flipped.items()))
    multi_ticker_clean = list(map(lambda x: min(x), multi_ticker.values()))

    cleaned_ticker_list = sorted(single_ticker_clean + multi_ticker_clean)

    stk_packet = cleaned_ticker_list[2500:2535]

    stock_comb_list = list(combinations(stk_packet, 10))

    data = pd.DataFrame()
    bad_stock = []

    for stock in stk_packet:
        try:
            logging.info('Reading Stock Data for ticker symbol : ' + stock)
            data[stock] = wb.DataReader(stock, data_source='yahoo', start='2019-12-30', end="2020-06-30")['Adj Close']
        except:
            bad_stock += stock
            logging.info('Issue with reading  : ' + stock)
            next

    data_fin_stg = data.fillna('')
    logging.info(data_fin_stg)

    num = 1
    start_time = time.time()

    for portfolio in stock_comb_list[0:1000]:
        pl = list(portfolio)
        actual_stks = [i for i in pl if i not in bad_stock]

        try:
            data_i = data_fin_stg[data_fin_stg.columns.intersection(actual_stks)]
            # print(data_i.head())
            logging.info("processing portfolio number : " + str(num))

            # data_i.to_csv('file_' + str(num) + '.csv', index=True, header=True)
            num = num + 1
            mu = expected_returns.mean_historical_return(data_i)
            S = risk_models.sample_cov(data_i)
            #
            ef = EfficientFrontier(mu, S)
            raw_weights = ef.max_sharpe()
            cleaned_weights = ef.clean_weights()

            result = dict(filter(lambda x: x[1] > 0.0, cleaned_weights.items()))
            result = dict(result)

            print("resultant dictionary : ", str(result))
            #
            #
            print(cleaned_weights)
            ef.portfolio_performance(verbose=True)
            print(raw_weights)
        except:
            logging.info("issue with " + data_i.columns)
            next

    logging.info("--- %s seconds ---" % (time.time() - start_time))


def getMultiStockName(tuples):
    flipped = {}
    for key, value in tuples.items():
        if value not in flipped:
            flipped[value] = [key]
        else:
            flipped[value].append(key)
    return flipped


if __name__ == "__main__":
    main()

    # shortLivedTck =data.columns[data.isnull().any()].tolist()
    # print(shortLivedTck)

    # l3 = list(set(cleanedTickerList) - set(shortLivedTck))
    # print(len(shortLivedTck))
    # print(len(cleanedTickerList))
    # print(len(l3))

# result = dict(filter(lambda x: x[1] > 0.0, cleaned_weights.items()))
# result = dict(result)

# print("resultant dictionary : ", str(result))
# ef.save_weights_to_file("weights.csv")  # saves to file
