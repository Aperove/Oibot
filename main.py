from telegram import Update
from telegram.ext import Updater, CommandHandler, CallbackContext
from telegram import BotCommand

import requests
import json
import pandas as pd
import os
from datagetter import * # 导入kira给的datagetter

# MPF and PLT
from datetime import datetime
import mplfinance as mpf
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

# DTW
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from fastdtw import fastdtw
import tempfile


folder_name_base = f"/mnt/data/swap/symbol_data"

def hello(update: Update, context: CallbackContext) -> None:
    update.message.reply_text(f'Hello, {update.effective_user.first_name}!')

pc_error = """格式错误
/pc 周期 y轴k线根数 x轴k线根数
/pc 1h 24 72 （3天与1天价格变动）
/pc 1d 7 30 （30天与7天价格变动）
"""



def pc(update: Update, context: CallbackContext) -> None:
    try:
        # period 15m 1h的周期
        period, kline_1, kline_2 = context.args 
        # 需要+1来保证今日昨日 多一根K线
        max_kline = max(kline_1, kline_2)
        min_kline = min(kline_1, kline_2)
        max_kline = -(int(max_kline) + 1)
        min_kline = -(int(min_kline) + 1)
    except ValueError:
        update.message.reply_text(pc_error)
        return
    
    # 从交换市场获取最新的数据
    try:
        update.message.reply_text('尝试获取数据')
        df = build_data(period, "swap", folder_name_base)
    except:
        update.message.reply_text('获取数据失败')
        return
    

    # 初始化一个空字典来存放结果
    data_1 = {}
    data_2 = {}


    # 遍历df字典
    for key in df:
        # 对每个值（数据框）取最后10行，并存入新的字典
        data_1[key] = df[key].iloc[max_kline:]
    for key in df:
        # 对每个值（数据框）取最后10行，并存入新的字典
        data_2[key] = df[key].iloc[min_kline:]


    # 计算价格变化
    price_changes_1 = {} 
    price_changes_2 = {} 

    # 对于每一个交易对，获取价格，计算涨幅
    for pair, dataframe in data_1.items():
        prices = dataframe['close'].tolist()
        change = (prices[-1] - prices[0]) / prices[0]
        price_changes_1[pair] = change

    for pair, dataframe in data_2.items():
        prices = dataframe['close'].tolist()
        change = (prices[-1] - prices[0]) / prices[0]
        price_changes_2[pair] = change

    volume_1 = {} 
    volume_2 = {} 

    # 对于每一个交易对，获取交易量
    for pair, dataframe in data_1.items():
        volume = dataframe['volume'].sum()
        volume_1[pair] = volume

    for pair, dataframe in data_2.items():
        volume = dataframe['volume'].sum()
        volume_2[pair] = volume



    # 制作散点图
    fig, ax = plt.subplots(figsize=(24,12))

    ax.xaxis.set_major_formatter(mtick.PercentFormatter(1))
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1))

    # 使用volume平均值来改变散点大小，乘以一定的系数让其更明显
    ax.scatter(price_changes_1.values(), price_changes_2.values(), s=[(v / (sum(volume_1.values()) / len(volume_1))) * 100 for v in volume_1.values()])

    for i, txt in enumerate(price_changes_1.keys()):
        ax.annotate(txt, (list(price_changes_1.values())[i], list(price_changes_2.values())[i]))

    # 移动坐标轴至中心位置
    ax.spines['left'].set_position('center')
    ax.spines['bottom'].set_position('center')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')

    avg_7d = np.mean(list(price_changes_1.values()))
    avg_2d = np.mean(list(price_changes_2.values()))
    # ax.axhline(y=avg_2d, color='k', linestyle='--', linewidth=1)
    # ax.axvline(x=avg_7d, color='k', linestyle='--', linewidth=1)


    plt.savefig('pc.png')

    with open("pc.png", "rb") as photo:
        context.bot.send_photo(chat_id=update.effective_chat.id, photo=photo)



dev_error = """格式错误
/dev 周期 EMA
/dev 5m 55 （对比5m周期EMA55 close和ema的偏离度）
/dev 1h 10 （对比1h周期EMA10 close和ema的偏离度）
"""




def dev(update: Update, context: CallbackContext) -> None:
    try:
        #period 15m 1h的周期 limit = ema
        period, ema_period = context.args 
        ema_period = int(ema_period)
    except ValueError:
        update.message.reply_text(dev_error)
        return
    
    # 从交换市场获取最新的数据
    try:
        update.message.reply_text('尝试获取数据')
        df = build_data(period, "swap", folder_name_base)
    except:
        update.message.reply_text('获取数据失败')
        return


    for symbol, data in df.items():
        data['EMA1'] = data['close'].ewm(span=ema_period, adjust=True).mean()
        # 偏离度计算
        data['DevEMA1'] = (data['close'] - data['EMA1']) / data['EMA1']

    # 通过plt.figure()创建一个figure对象，并设置长宽比例
    fig, ax = plt.subplots(figsize=(16, 13))

    # 假设dfs是一个字典，包含了所有的dataframe，其中键是标的名称（如'BTCUSDT'），值是对应的dataframe
    for symbol, df in df.items():
        x = df['candle_begin_time']
        y = df['DevEMA1']
        
        # 把时间作为x轴，'DevEMA1'作为y轴
        ax.plot(x, y, label=symbol, linewidth=1)
        
        # 在每条线的尾部添加标签
        ax.text(x.iloc[-1], y.iloc[-1], symbol, fontsize=10, verticalalignment='bottom')

    # 设置图的元信息
    ax.set_xlabel('Time')
    ax.set_ylabel('Deviation')
    ax.set_title('Dev')

    # 保存
    plt.savefig('dev.png')

    with open("dev.png", "rb") as photo:
        context.bot.send_photo(chat_id=update.effective_chat.id, photo=photo)



dtw_error = """格式错误
/dtw 标的 周期 根数
/dtw btcusdt 15m 100（对比15m周期100根K线内 其他标的与BTC的相似性）
/dtw jasmyusdt 5m 50（对比5m周期50根K线内 其他标的与JASYMY的相似性）
"""


# DTW + EMA趋势
def dtw(update: Update, context: CallbackContext) -> None:

    try:
        # symbol = 作为base的symbol
        # period = 周期 15m 1h 等等
        symbol, period, limited = context.args
        symbol = symbol.upper()
        limited = -int(limited)
    except ValueError:
        update.message.reply_text(dtw_error)
        return


    # 从交换市场获取最新的数据
    try:
        update.message.reply_text('尝试获取数据')
        df = build_data(period, "swap", folder_name_base)
    except:
        update.message.reply_text('获取数据失败')
        return
    

    # 执行特征工程
    for symbol, data in df.items():
        # 用收盘价的百分比变化计算'相对变化'
        data['relative_change'] = data['close'].pct_change()

    # 数据正则化
    scaler = MinMaxScaler()

    for symbol, data in df.items():
        # 将'相对变化'的值进行归一化处理，并将Nan值填充为0
        data['relative_change'] = scaler.fit_transform(data[['relative_change']].fillna(0)).flatten()

    # 添加额外的EMA指标
    period_1 = 12  # EMA1 如果接针需要小周期
    period_2 = 55  # EMA2
    period_3 = 120  # EMA3

    for symbol, data in df.items():
        data['EMA1'] = data['close'].ewm(span=period_1, adjust=True).mean()
        data['EMA2'] = data['close'].ewm(span=period_2, adjust=True).mean()
        data['EMA3'] = data['close'].ewm(span=period_3, adjust=True).mean()

    for symbol, data in df.items():
        data['EMA Long short'] = np.where((data['EMA1'] > data['EMA3']), 1, np.nan) # EMA1 > 2 > 3    & (data['EMA1'] > data['EMA3'])

    # 获取基准符号数据
    base_symbol_data = df[symbol]['relative_change'].tolist()

    dtw_distances = {}


    for symbol, data in list(df.items()):
        comparison_symbol_data = data['relative_change'].tolist()
        # 使用fastdtw函数计算两个时间序列之间的动态时间扭曲（DTW）距离
        distance, _ = fastdtw(base_symbol_data[limited:], [float(i) for i in comparison_symbol_data][limited:])
        dtw_distances[symbol] = [symbol, distance, 'LONG' if data['EMA Long short'].iloc[-1] == 1 else '']

    # 将字典转换为数据框
    dtw_distances_df = pd.DataFrame(list(dtw_distances.values()), columns=['Symbol', 'DTW_distance', 'EMA_long_short'])

    # 按照DTW_distance对数据框进行排序
    sorted_dtw_distances_df = dtw_distances_df.sort_values(by='DTW_distance')
    file = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
    sorted_dtw_distances_df.to_csv(file.name, index=False)


    # 打印排序好的数据框
    try:
        # 在 Telegram 中发送文件
        with open(file.name, 'rb') as f:
            context.bot.send_document(chat_id=update.effective_chat.id, document=f)
    finally:
        # 确保在文件发送后删除临时文件
        os.unlink(file.name)



def oi(update: Update, context: CallbackContext) -> None:

    try:
        symbol, interval = context.args
    except:
        update.message.reply_text('格式错误')
        return
    

    
    if "!" in symbol:
        symbol = symbol.replace('!', '')
    else:
        symbol = symbol + 'usdt'
        

    data = {
        'symbol': symbol,
        'pair': symbol,
        'period': interval,
        'interval': interval,
        'limit':'150'
    }



    # 给参数赋值
    symbol = 'btcusdt'
    interval = '1h'

    # 将参数放入字典
    data = {
        'symbol': symbol,
        'pair': symbol,
        'period': interval,
        'interval': interval,
        'limit': '150'
    }

    # 定义各个请求的URL
    kline_url = 'https://fapi.binance.com/fapi/v1/klines'
    lsur_url = 'https://fapi.binance.com/futures/data/globalLongShortAccountRatio'
    oi_url = 'https://fapi.binance.com/futures/data/openInterestHist'
    fr_url = 'https://fapi.binance.com/fapi/v1/premiumIndex'

    # 从API获取数据
    kline_json = requests.get(kline_url, params=data).json()
    lsur_data = requests.get(lsur_url, params=data).json()
    oi_data = requests.get(oi_url, params=data).json()
    fr_data = requests.get(fr_url, params=data).json()

    # 处理kline数据，转换时间戳，将数据放到DataFrame
    tmp = []
    pair = []
    for base in kline_json:
        tmp = []
        for i in range(0,6):
            if i == 0:
                base[i] = datetime.fromtimestamp(base[i]/1000)
            tmp.append(base[i])
        pair.append(tmp)

    # 生成数据帧
    df = pd.DataFrame(pair, columns=['date', 'open', 'high', 'low', 'close', 'volume'])
    df.date = pd.to_datetime(df.date)
    df.set_index("date", inplace=True)
    df = df.astype(float)

    # 数据帧的列转换为浮点值型
    lsur_df = pd.DataFrame(lsur_data)
    lsur_df['longShortRatio'] = lsur_df['longShortRatio'].astype(float)

    oi_df = pd.DataFrame(oi_data)
    oi_df['sumOpenInterest'] = oi_df['sumOpenInterest'].astype(float)
    oi_df['sumOpenInterestValue'] = oi_df['sumOpenInterestValue'].astype(float)

    # 计算最后一个资料费率
    last_fr = float(fr_data['lastFundingRate'])
    last_fr = str(f'{last_fr:.4%}')

    # 定义颜色
    my_color = mpf.make_marketcolors(
        edge='black',
        wick='black',
    )
    # 设定风格
    style_w = mpf.make_mpf_style(marketcolors=my_color, gridcolor='(0.82, 0.83, 0.85)')

    # 获取一些特定的值
    last_lusr = str(lsur_df.iloc[[-1],[2]].values[0][0])
    last_oi = str(oi_df.iloc[[-1],[1]].values[0][0])
    last_oival = str(oi_df.iloc[[-1],[2]].values[0][0])
    last_fr = str(last_fr)

    # 组装一条消息
    message_data = "当前的多空比为:" + last_lusr + "。 开仓量（张）为:" + last_oi + "。 开仓量（U）为:" + last_oival + "。 资金费率为: " + last_fr

    # 设置mplfinance绘图参数
    index = [
        mpf.make_addplot(lsur_df['longShortRatio'], panel=1, color='red', alpha=0.7),
        mpf.make_addplot(oi_df['sumOpenInterest'], panel=1, color='black', alpha=0.4),
        mpf.make_addplot(lsur_df['longShortRatio'], panel=2, color='red', alpha=0.7),
        mpf.make_addplot(oi_df['sumOpenInterestValue'], panel=2, color='green', alpha=0.4)
    ]

    # 绘制并保存图表
    mpf.plot(
        df, type='candle',
        style=style_w,
        addplot=index,
        figsize=(16, 9),
        ylabel='Price',
        savefig="oi.png"
    )

    update.message.reply_text(message_data)
    with open("oi.png", "rb") as photo:
        context.bot.send_photo(chat_id=update.effective_chat.id, photo=photo)




def main() -> None:
    # 创建一个更新程序实例，需要将其中的字符串替换成你自己的Telegram Bot的token
    updater = Updater("your token", use_context=True)

    # 获取更新程序的调度程序以注册处理器
    dispatcher = updater.dispatcher

    # 设置你的bot的命令及其说明
    commands = [
        BotCommand('oi', '多空比 '),
        BotCommand('dtw', 'DTW 数据间相似度 格式：标的 周期 根数'),
        BotCommand('dev', 'EMA 偏离度 格式：周期 EMA'),
        BotCommand('pc', '价格变动 格式：周期 y轴根数 x轴根数'),
    ]
    updater.bot.set_my_commands(commands)

    # 添加一个命令处理器
    dispatcher.add_handler(CommandHandler("hello", hello))
    dispatcher.add_handler(CommandHandler("oi", oi))
    dispatcher.add_handler(CommandHandler("dtw", dtw))
    dispatcher.add_handler(CommandHandler("dev", dev))
    dispatcher.add_handler(CommandHandler("pc", pc))


    # 开始轮询
    updater.start_polling()

    # 运行机器人，直到你为程序发送一个停止信号（例如，通过按Control + C）。
    updater.idle()

if __name__ == '__main__':
    main()



