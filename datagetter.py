import ccxt
import pandas as pd
from datetime import datetime, timedelta
import time
from multiprocessing.dummy import Pool
import os
import gc
import shutil


SEVER_MODE = True #是否在服务器上运行，是——True；否——False

READ_CSV_CONTROL = True #CSV读取行数限制，是——True；否——False
CUT_NUMBER = 1000 #CSV读取限制行数
MAX_PROCESSES = 20 # 限制最大进程数

BINANCE_CONFIG = {
    'apiKey': '',
    'secret': '',
    'timeout': 3000,
    'rateLimit': 10,
    'enableRateLimit': False}
exchange = ccxt.binance()
#启动系统代理
if SEVER_MODE == False :
    exchange.https_proxy = 'http://127.0.0.1:7890/'

def get_future_symbols(exchange):
    print(f'正在获取future标的列表')
    future =exchange.fapiPublicGetExchangeInfo()
    binance_symbol_list = []
    for i in range(len(future['symbols'])):
        if future['symbols'][i]['status'] == 'TRADING':
            symbol_name = future['symbols'][i]['symbol']
            binance_symbol_list.append(symbol_name)
    print(f'future标的列表获取完毕，一共{len(binance_symbol_list)}个标的被获取到')
    return binance_symbol_list

def get_spot_symbols(exchange):
    print(f'正在获取spot标的列表')
    future =exchange.publicGetExchangeinfo()
    binance_symbol_list = []
    for i in range(len(future['symbols'])):
        quote_filter_list = ['BTC','ETH','BNB','SOL','USDT'] #对交易quote进行筛选
        quote_filter = future['symbols'][i]['quoteAsset'] in quote_filter_list
        if future['symbols'][i]['status'] == 'TRADING' and quote_filter:
            symbol_name = future['symbols'][i]['baseAsset']+'/'+future['symbols'][i]['quoteAsset']
            binance_symbol_list.append(symbol_name)
    print(f'spot标的列表获取完毕，一共{len(binance_symbol_list)}个标的被获取到')
    return binance_symbol_list


def build_data(period,data_type,folder_name_base):
    print(f'【{period}】正在进行计算数据构建整理')
    if data_type == 'spot':
        binance_symbol_list = get_spot_symbols(exchange)
    elif data_type == 'swap':
        binance_symbol_list = get_future_symbols(exchange)
    # Load the data
    csv_symbol_candle_data, missing_symbols_list = load_csv_to_dict(folder_name_base, binance_symbol_list,period)
    if len(missing_symbols_list)> 0:
        del csv_symbol_candle_data
        gc.collect()
        check_csv_data(folder_name_base,missing_symbols_list,period,data_type)
        csv_symbol_candle_data, missing_symbols_list = load_csv_to_dict(folder_name_base, binance_symbol_list,period)
    symbol_candle_data = update_saved_data(csv_symbol_candle_data, binance_symbol_list,missing_symbols_list, period, folder_name_base,data_type, if_now=True)
    del csv_symbol_candle_data
    gc.collect()
    symbol_candle_data = update_saved_data(symbol_candle_data, binance_symbol_list,missing_symbols_list, period, folder_name_base, data_type, if_now=True)
    symbol_candle_data = candle_data_cut(symbol_candle_data, 1000)
    print(f'计算数据构建整理完毕')
    return symbol_candle_data

# 获取单个币种的1小时数据
def fetch_binance_swap_candle_data(exchange, symbol, basic_period, limit):
    #print(datetime.now(), '开始获取k线数据：', symbol)

    # 获取数据
    try:
        kline = exchange.fapipublic_get_klines({'symbol': symbol, 'interval': basic_period, 'limit': limit})
        # Additional code to process kline data
        # 将数据转换为DataFrame
        columns = ['candle_begin_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_volume',
                   'trade_num',
                   'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore']

        df = pd.DataFrame(kline, columns=columns, dtype='float')
        # 整理数据
        df['candle_begin_time'] = pd.to_datetime(df['candle_begin_time'], unit='ms') + pd.Timedelta(hours=8)  # 时间转化为东八区
        df['symbol'] = symbol  # 添加symbol列
        columns = ['symbol', 'candle_begin_time', 'open', 'high', 'low', 'close', 'volume', 'quote_volume','trade_num',
                   'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume']
        df = df[columns]

        #     #输出测试信息
        #     if df.iloc[-1]['candle_begin_time'] == check_time :
        #         print('获取错误', df.iloc[-1]['candle_begin_time'])

        #print(datetime.now(), '结束获取k线数据：', symbol, df.iloc[-1]['candle_begin_time'])
        time.sleep(1)
    except Exception as e:
        df = pd.DataFrame()
        print("--------------------------------------------------")
        print(symbol, f"发生异常：{e}")
        print("--------------------------------------------------")

    return symbol, df


# 并行获取所有币种永续合约数据的1小时K线数据
def fetch_all_binance_swap_candle_data(exchange, symbol_list, basic_period, LIMIT):
    """
    并行获取所有币种永续合约数据的1小时K线数据
    :param exchange:
    :param symbol_list:

    :return:
    """
    print('开始获取所有币种K线数据')
    # 创建参数列表

    arg_list = [(exchange, symbol, basic_period, LIMIT) for symbol in symbol_list]
    # 多进程获取数据
    s_time = time.time()
    print(len(arg_list))

    with Pool(processes=min(len(arg_list), MAX_PROCESSES)) as pl:
        # 利用starmap启用多进程信息
        result = pl.starmap(fetch_binance_swap_candle_data, arg_list)

    df = dict(result)
    print('获取所有币种K线数据完成，花费时间：', time.time() - s_time)
    return df



def fetch_binance_spot_candle_data(exchange, symbol, basic_period, limit):
    """
    通过ccxt的接口fapiPublic_get_klines，获取永续合约k线数据
    获取单个币种的1小时数据
    :param exchange:
    :param symbol:
    :param limit:
    :return:
    """
    #print(datetime.now(), '开始获取k线数据：', symbol)
    try:
        kline = exchange.fetch_ohlcv(symbol, basic_period, since=None, limit=limit)
        # Additional code to process kline data
        # 将数据转换为DataFrame
        columns = ['candle_begin_time', 'open', 'high', 'low', 'close', 'volume']

        df = pd.DataFrame(kline, columns=columns, dtype='float')
        # 整理数据
        df['candle_begin_time'] = pd.to_datetime(df['candle_begin_time'], unit='ms') + pd.Timedelta(hours=8)  # 时间转化为东八区
        df['symbol'] = symbol  # 添加symbol列
        columns = ['symbol', 'candle_begin_time', 'open', 'high', 'low', 'close', 'volume']
        df = df[columns]

        #     #输出测试信息
        #     if df.iloc[-1]['candle_begin_time'] == check_time :
        #         print('获取错误', df.iloc[-1]['candle_begin_time'])

        #print(datetime.now(), '结束获取k线数据：', symbol, df.iloc[-1]['candle_begin_time'])
        time.sleep(1)
    except Exception as e:
        df = pd.DataFrame()
        print("--------------------------------------------------")
        print(symbol, f"发生异常：{e}")
        print("--------------------------------------------------")

    return symbol, df


# 并行获取所有币种永续合约数据的1小时K线数据
def fetch_all_binance_spot_candle_data(exchange, symbol_list,  basic_period, LIMIT):
    print('开始获取所有币种K线数据')
    # 创建参数列表
    arg_list = [(exchange, symbol,  basic_period, LIMIT) for symbol in symbol_list]
    # 多进程获取数据
    s_time = time.time()
    print(len(arg_list))
    with Pool(processes=min(len(arg_list), MAX_PROCESSES)) as pl:
        # 利用starmap启用多进程信息
        result = pl.starmap(fetch_binance_spot_candle_data, arg_list)

    df = dict(result)
    print('获取所有币种K线数据完成，花费时间：', time.time() - s_time)
    return df



# 创建一个映射字典，键为消息类型，值为对应的处理函数
candle_data_handlers = {
    'spot': fetch_all_binance_spot_candle_data,
    'swap': fetch_all_binance_swap_candle_data,
}




# Updated function to include the candle period in the folder and file names
def save_symbol_data_as_csv_with_period(symbol_candle_data, folder_name_base, period):

    print(f"开始保存数据到CSV文件")
    # Format the folder name to include the period
    folder_name = f"{folder_name_base}_{period}"
    # Create the folder if it doesn't exist
    os.makedirs(folder_name, exist_ok=True)
    saved_symbol_list = []
    # Iterate through the symbols and save their data
    for symbol, data in symbol_candle_data.items():
        df = pd.DataFrame(data)
        if len(df)>0:
            folder_symbol_name = symbol.replace("/", "_")
            csv_filename = os.path.join(folder_name, f"{folder_symbol_name}.csv")
            df.to_csv(csv_filename, index=False)
            saved_symbol_list.append(symbol)
        else:
            print(f"获取到的数据为空{symbol}")
    print(f"Data of {period}saved to {folder_name}，一共 {len(saved_symbol_list)}个标的数据被保存")


def load_csv_to_dict(folder_name_base, symbols, period):
    print(f"开始加载CSV文件到数据")
    symbol_candle_data = {}
    missing_symbols = []
    folder_name = f"{folder_name_base}_{period}"
    print(folder_name)
    for symbol in symbols:
        folder_symbol_name = symbol.replace("/", "_")
        csv_filename = os.path.join(folder_name, f"{folder_symbol_name}.csv")

        # Check if the file exists
        if os.path.exists(csv_filename):
            # Read the CSV file and add to the dictionary

            df = pd.read_csv(csv_filename)
            df['candle_begin_time'] = pd.to_datetime(df['candle_begin_time'])
            #这里设置可选项目,如果需要控制CSV的输入K线根数
            if READ_CSV_CONTROL == True:
                df = candle_data_cut_of_dataframe(df, CUT_NUMBER)
            symbol_candle_data[symbol] = df
        else:
            # Add to the list of missing symbols
            missing_symbols.append(symbol)
    print(f"CSV文件加载完毕，共有{len(missing_symbols)}个symbol丢失")
    return symbol_candle_data, missing_symbols

#获取K线数据，最大尝试三次
def get_kline_data(exchange, binance_symbol_list,  period, kline_Limit,data_type):
    print(f'现在开始获取K线数据')
    max_retry_times = 3
    retry_times = 1
    # 直接从映射中获取函数对象
    fetch_candle_data = candle_data_handlers.get(data_type)


    while retry_times <= max_retry_times:
        print(f'正在进行第{retry_times}次获取')
        if retry_times == 1:
            symbol_candle_data = fetch_candle_data(exchange, binance_symbol_list,
                                                                    basic_period=period, LIMIT=kline_Limit)
            filed_symbol_list = []

        else:
            retry_symbol_candle_data = fetch_candle_data(exchange, filed_symbol_list,
                                                                          basic_period=period, LIMIT=kline_Limit)
            filed_symbol_list = []
            retry_symbol_candle_data = {k: v for k, v in retry_symbol_candle_data.items() if not v.empty}

            for symbol in retry_symbol_candle_data:
                if symbol in symbol_candle_data:
                    symbol_candle_data[symbol] = retry_symbol_candle_data[symbol]
                    print(f"{symbol}数据再获取成功")
            del retry_symbol_candle_data
            gc.collect()
        for symbol in symbol_candle_data.keys():
            if len(symbol_candle_data[symbol]) == 0:
                filed_symbol_list.append(symbol)
        retry_times += 1
        if len(filed_symbol_list) == 0:
            print("【SCUCCESS】所有补充数据获取成功")
            break
    return symbol_candle_data



def update_saved_data(csv_symbol_candle_data, binance_symbol_list,missing_symbol_list, period, folder_name_base,data_type, if_now=False):
    print(f'现在开始获取K线数据')
    # 定义时间
    first_key_list_method = list(csv_symbol_candle_data)[0]
    start_time = csv_symbol_candle_data[first_key_list_method]["candle_begin_time"].iloc[-2]
    end_time = csv_symbol_candle_data[first_key_list_method]["candle_begin_time"].iloc[-1]
    #去除获取失败的symbol
    if len(missing_symbol_list)>0:
        binance_symbol_list = [item for item in binance_symbol_list if item not in missing_symbol_list]
    # 计算时间间隔a
    time_interval_a = end_time - start_time

    # hours=1当前时间UTC+9使用,hours=0当前时间UTC+8使用
    if SEVER_MODE == True:
        current_time = datetime.now() - timedelta(hours=1)
    else:
        current_time = datetime.now()

    # 计算从2023-11-11 19:00:00到当前时间的间隔
    time_interval_to_now = current_time - end_time

    # 计算离当前时间有多少个时间间隔a
    number_of_intervals = time_interval_to_now / time_interval_a
    print(f'当前数据周期{period},与数据差值为{number_of_intervals}')
    if number_of_intervals<1:
        print('当前时间周期已为最新数据')
        symbol_candle_data = csv_symbol_candle_data.copy()
        return symbol_candle_data
    else:
        if (number_of_intervals-1)<990:
            print('现在开始获取增量数据到最新')
            add_Limit = int(number_of_intervals+1)
            kline_Limit = add_Limit
            symbol_candle_data = get_kline_data(exchange, binance_symbol_list, period,kline_Limit,data_type)
            print('开始合并数据')
            for symbol in symbol_candle_data:
                hist_df = csv_symbol_candle_data[symbol]
                new_df = symbol_candle_data[symbol]
                spilt_timepoint = hist_df["candle_begin_time"].iloc[-1]
                hist_df_filtered = hist_df[hist_df['candle_begin_time'] < spilt_timepoint]
                new_df_filtered = new_df[new_df['candle_begin_time'] >= spilt_timepoint]
                symbol_candle_data[symbol] = pd.concat([hist_df_filtered , new_df_filtered]).drop_duplicates().reset_index(drop=True)
            print('保存最新数据')
            save_symbol_data_as_csv_with_period(symbol_candle_data, folder_name_base, period)
            return symbol_candle_data
        else: # 修改过的部分 重新获取一次
            print('数据缺失长度超出1000，正在删除旧文件并重新获取数据,缺失K线根数', number_of_intervals)

            # 删除旧文件
            folder_path = os.path.join(os.getcwd(), folder_name_base)
            for file_name in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file_name)
                if os.path.isfile(file_path):
                    os.remove(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)

            # 重新获取数据
            symbol_candle_data = get_kline_data(exchange, binance_symbol_list, period, 1000, data_type)
                
            print('保存最新数据')
            save_symbol_data_as_csv_with_period(symbol_candle_data, folder_name_base, period)
                
            return symbol_candle_data


#查找并获取缺失数据
def check_csv_data(folder_name_base,missing_symbols_list,period,data_type):
    # Check if any symbols are missing
    if missing_symbols_list:
        print("开始重新获取缺失数据")
        missing_symbol_limit = 1000
        missing_symbol_data = get_kline_data(exchange, missing_symbols_list, period,missing_symbol_limit,data_type)
        save_symbol_data_as_csv_with_period(missing_symbol_data, folder_name_base, period)
    else:
        print("All symbols loaded successfully.")


#切分蜡烛图数据，主要为了控制计算时间以及图表显示等等
def candle_data_cut(symbol_candle_data,cut_num=1000):
    print(f'开始截取数据，截取长度为{cut_num}')
    new_symbol_candle_data = {}
    for symbol in symbol_candle_data.keys():
        df_data = candle_data_cut_of_dataframe(symbol_candle_data[symbol],cut_num)
        new_symbol_candle_data[symbol] = df_data
    print('已完成数据截取')
    return new_symbol_candle_data

def candle_data_cut_of_dataframe(df,cut_num):
    df_data = df.tail(cut_num).copy()
    df_data.sort_values(by=['candle_begin_time'], inplace=True)
    df_data.drop_duplicates(subset=['candle_begin_time'], inplace=True)
    df_data.reset_index(inplace=True, drop=True)
    return df_data
