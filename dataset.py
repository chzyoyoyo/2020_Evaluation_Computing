import numpy as np
import csv
import time
import indicator

def GetDataset(path):
    with open(path, encoding='utf-8-sig') as f:
        myData = csv.DictReader(f)
        result = []
        for data in myData:
            timestamp = TimeStampProcessing(data["日期"])
            close_price = NumericStringProcessing(data["收市"])
            open_price = NumericStringProcessing(data["開市"])
            highest = NumericStringProcessing(data["高"])
            lowest = NumericStringProcessing(data["低"])
            volume = VolumeProcessing(data["成交量"])
            result.append([timestamp,close_price , open_price , highest , lowest,volume])
        result = np.array(result)
    return result

def TimeStampProcessing(date): #2019年5月19號 -> 20190519 (int)
    timestamp = time.strptime(date,'%Y年%m月%d日')
    timestamp = float(time.strftime('%Y%m%d' , timestamp))
    return timestamp

def NumericStringProcessing(str):
    nums = str.split(',')
    result = 0
    for num in nums:
        result *= 1000
        result += float(num)
    return result

def VolumeProcessing(str):
    result = 0
    if str[-1] is 'M':
        result = float(str[:-1])*1000000
    elif str[-1] is 'K':
        result = float(str[:-1])*1000
    elif str[-1] is '-':
        result = 0.0
    else:
        result = float(str)
    return result

def TransformToBinary(dataset):
    # data: 1 ~ -31 days, every day will compare to previous 1 day (5) and 30 days (5) and next 7 day (1).
    # Hence, binary string will be (N-30-7)x(5+5+1) array
    size = dataset.shape[0]-37
    binary_string = np.zeros((size , 11) , dtype=np.int)

    for i in range(size):
        index = i + 7 # get the index of dataset

        thisday = dataset[index]
        nextday = dataset[index-7] #next day
        prvday = dataset[index+1] #previous day
        monthdata = dataset[index+1:index+31] #previous 30 days
        averagedata = np.sum(monthdata ,axis= 0)/30 #average of previous 30 days

        binary_string[i][:5] = thisday[1:] > prvday[1:] #represent the up and downs compared to previous day
        binary_string[i][5:10] = thisday[1:] > averagedata[1:] #represent the up and downs compared to prvious 30 days
        binary_string[i][-1] = nextday[1] > thisday[1]  # up and down

    arr1 = indicator.probability(dataset, 30).reshape(-1,1)
    arr2_1 = indicator.comAvg(dataset, 10, 5).reshape(-1,1)
    arr2_2 = indicator.comAvg(dataset, 30, 10).reshape(-1,1)
    arr2_3 = indicator.comAvg(dataset, 30, 7).reshape(-1,1)
    arr2 = np.hstack((arr2_1, arr2_2, arr2_3))

    arr3 = indicator.RSI(dataset)

    # print(arr1.shape)

    # print(arr2.shape)
    # print(arr3.shape)
    # print(binary_string.shape)

    input_string = np.hstack((arr1, arr2, arr3, binary_string))

    #print(input_string.shape)

    return input_string

def TransformToBinary2(dataset , ref_days = 30 , pred_days = 7):
    size = dataset.shape[0] - ref_days - pred_days
    binary_string = np.zeros((size , ref_days + 1) , dtype = np.int)

    for i in range(size):
        index = i + pred_days

        for j in range(ref_days):
            anchor__day_price = dataset[index + j][1] #close price
            prv_day_price = dataset[index + j + 1][1] #close price
            binary_string[i][j] = anchor__day_price > prv_day_price

        binary_string[i][-1] = dataset[index - pred_days][1] > dataset[index][1]

    return binary_string