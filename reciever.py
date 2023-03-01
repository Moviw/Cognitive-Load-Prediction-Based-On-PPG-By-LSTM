import serial
import torch
import msvcrt
from serial.tools import list_ports
import time
import numpy as np
import matplotlib.pyplot as plt
from utils.feature_extraction import find_peak_trough, feature_extraction
from sklearn.preprocessing import MinMaxScaler
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

SEQ_LENGTH = 3
INPUT_SIZE = 6


def read_single_data(ser):
    # 读取数据函数
    return int(ser.read().hex(), 16)


def create_ser():
    # 定义串口
    print('''************\n正在创建串口\n************\n''')
    ser = serial.Serial()
    ser.port = serials[0].name
    ser.baudrate = 9600
    ser.bytesize = 8
    ser.open()
    print('''************\n创建串口完成\n************\n''')
    return ser


def get_fea_matrix(ser):
    scaler = MinMaxScaler()
    count = 0
    data_list = []
    while True:
        data = read_single_data(ser)
        # print("十进制数为:%d" % (data))
        data_list.append(data)
        count += 1
        if count % 300 == 0:
            data_ndarray = np.array(data_list)
            peaks, troughs = find_peak_trough(data_ndarray, distance=50)

            f1, f2, f3, f4, f5, f8, f9, f10 = feature_extraction(
                data_ndarray, peaks, troughs)
            if(len(f1) >= SEQ_LENGTH):
                # 拿最新的SEQ_LENGTH个
                fea_matrix = scaler.fit_transform(np.hstack(
                    (f1, f3, f4, f5, f8, f9))[-SEQ_LENGTH:, :]).reshape(1, SEQ_LENGTH, INPUT_SIZE)
                return torch.tensor(fea_matrix).to(device).float()


if __name__ == '__main__':
    # 当前所有的串口设备
    serials = list(list_ports.comports())
    print([i.name for i in serials])

    ser = create_ser()
    lstm = torch.load('model/lstm.pkl')
    while True:
        # if ord(msvcrt.getch()) in [0x20, 81, 113]:  # 按下空格、q、Q结束
        #     break
        fea_matrix = get_fea_matrix(ser)
        output = lstm(fea_matrix)

        category = torch.argmax(output).cpu().numpy()  # 输出判断结果 0为正常 1为负荷
        print("current status : NORM" if category ==
              0 else 'current status : LOADED')
        # if category == 0:
        #     print("status: norm")
        # else:
        #     print("status: loaded")
    ser.close()
