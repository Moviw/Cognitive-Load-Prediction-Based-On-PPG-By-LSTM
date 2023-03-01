import numpy as np
# from fft import fft_ppg
from scipy import signal
import matplotlib.pyplot as plt

distance = 50

# 找波峰波谷


def find_peak_trough(data, distance):
    peaks = signal.find_peaks(data, distance=distance)
    # print("peaks are ", peaks[0])

    troughs = signal.find_peaks(-1*data, distance=distance)
    # print("troughes are ", troughs[0])
    return peaks[0], troughs[0]


def get_der(data):
    # 求导(离散版)
    return np.array([data[i+1]-data[i] for i in range(len(data)-1)])


def peak_trough_subdivide(data, peaks, troughs):
    peak_up = np.array([peaks[i] for i in range(len(peaks)-1)
                        if data[peaks[i]] > data[peaks[i+1]]])

    peak_down = np.sort(
        np.array(list(set(peaks.tolist())-set(peak_up.tolist()))))

    trough_up = np.array([troughs[i] for i in range(len(troughs)-1)
                          if data[troughs[i]] > data[troughs[i+1]]])

    trough_down = np.sort(
        np.array(list(set(troughs.tolist())-set(trough_up.tolist()))))
    return peak_up, peak_down, trough_up, trough_down


def plot_peak_trough(data, peaks, troughs, c='k'):
    plt.plot(data, c=c)
    for i in range(len(peaks)):
        plt.plot(peaks[i], data[peaks[i]], '^', markersize=10)

    for i in range(len(troughs)):
        plt.plot(troughs[i], data[troughs[i]], 'v', markersize=10)
    plt.show()


def get_sVRI(data, peak_up, trough_down):
    """
        根据找到的波峰波谷再根据公式计算出每个周期内的sVRI
        return : sVRI列表
    """

    sVRI = []

    for i in range(len(trough_down)-1):
        sum1 = 0
        sum2 = 0
        sum1 = np.sum(data[trough_down[i]:peak_up[i]])
        sum2 = np.sum(data[peak_up[i]:trough_down[i+1]])
        sVRI.append(1.0*sum1/sum2)

    sum1 = np.sum(data[trough_down[-1]:peak_up[-1]])
    sum2 = np.sum(data[peak_up[-1]:-1])
    sVRI.append(1.0*sum1/sum2)

    return np.array(sVRI).reshape(-1, 1)


def get_t(peak_up, peak_down, trough_up, trough_down):
    t1 = []
    for i in range(len(trough_down)):
        for j in range(len(peak_up)):
            if(peak_up[j] > trough_down[i]):
                t1.append(peak_up[j]-trough_down[i])
                break
    t1 = np.array(t1)

    t2 = []
    for i in range(len(trough_down)):
        for j in range(len(trough_up)):
            if(trough_up[j] > trough_down[i]):
                t2.append(trough_up[j]-trough_down[i])
                break
    t2 = np.array(t2)

    t3 = []
    for i in range(len(trough_down)):
        for j in range(len(peak_down)):
            if(peak_down[j] > trough_down[i]):
                t3.append(peak_down[j]-trough_down[i])
                break
    t3 = np.array(t3)

    t_pi = np.array([trough_down[i+1]-trough_down[i]
                    for i in range(len(trough_down)-1)])
    t_pi = np.append(t_pi, (trough_down[-1]-trough_down[0])/len(trough_down))

    return t1, t2, t3, t_pi


# 由于获得的波峰波谷不保证一定精准 所以需要这里只选择成对并按顺序出现的波峰波谷
def wash_peak_trough(peak_up, peak_down, trough_up, trough_down):
    washed_peak_up, washed_peak_down, washed_trough_up, washed_trough_down = [], [], [], []
    min_sequence_num = min(
        peak_up.shape[0], peak_down.shape[0], trough_down.shape[0], trough_up.shape[0])
    for i in range(min_sequence_num):
        if(i != min_sequence_num-1):
            temp_peak_up = np.where(
                (peak_up > trough_down[i]) & (peak_up < trough_down[i+1]))
            temp_peak_down = np.where(
                (peak_down > trough_down[i]) & (peak_down < trough_down[i+1]))
            temp_trough_up = np.where(
                (trough_up > trough_down[i]) & (trough_up < trough_down[i+1]))

        else:
            temp_peak_up = np.where(
                (peak_up > trough_down[i]))
            temp_peak_down = np.where(
                (peak_down > trough_down[i]))
            temp_trough_up = np.where(
                (trough_up > trough_down[i]))

        # print("temp_peak_up:", peak_up[temp_peak_up])
        # print('temp_peak_down:', peak_down[temp_peak_down])
        # print('temp_trough_up:', trough_up[temp_trough_up])
        if(len(peak_up[temp_peak_up[0]]) == 0 or len(peak_down[temp_peak_down[0]]) == 0 or len(trough_up[temp_trough_up[0]]) == 0):
            continue
        washed_peak_up.append(peak_up[temp_peak_up[0]][0])  # 防止有多个
        washed_peak_down.append(peak_down[temp_peak_down[0]][0])  # 防止有多个
        washed_trough_up.append(trough_up[temp_trough_up[0]][0])  # 防止有多个
        washed_trough_down.append(trough_down[i])

    return np.array(washed_peak_up), np.array(washed_peak_down), np.array(washed_trough_up), np.array(washed_trough_down)


def feature_extraction(data, peaks, troughs):
    peak_up, peak_down, trough_up, trough_down = peak_trough_subdivide(
        data, peaks, troughs)
    # print("peak_up,peak_down,trough_up,trough_down:{}{}{}{}".format(
    #     peak_up.shape, peak_down.shape, trough_up.shape, trough_down.shape))

    peak_up, peak_down, trough_up, trough_down = wash_peak_trough(
        peak_up, peak_down, trough_up, trough_down)

    if (len(peak_up) == 0 or len(peak_down) == 0 or len(trough_up) == 0 or len(trough_down) == 0):
        # print("there")
        return [], [], [], [], [], [], [], []

    x = np.array(data[peak_up])
    y = np.array(data[peak_down])
    z = np.array(data[trough_up])
    # print("peak_up,peak_down,trough_up,trough_down:{}{}{}{}".format(
    #     peak_up.shape, peak_down.shape, trough_up.shape, trough_down.shape))

    t1, t2, t3, t_pi = get_t(peak_up, peak_down, trough_up, trough_down)
    # print("t1,t2,t3,t_pi:{}{}{}{}".format(
    #     t1.shape, t2.shape, t3.shape, t_pi.shape))

    der1 = get_der(data)
    der1_peak, der1_trough = find_peak_trough(der1, distance)

    der2 = get_der(der1)
    der2_peak, der2_trough = find_peak_trough(der2, distance)

    # plot_peak_trough(der1, der1_peak, der1_trough, 'r')
    # plot_peak_trough(der2, der2_peak, der2_trough, 'b')
    # plt.show()

    a1_pos, e1_pos, f1_pos, b1_pos = peak_trough_subdivide(
        der1, der1_peak, der1_trough)
    a1_pos, e1_pos, f1_pos, b1_pos = wash_peak_trough(
        a1_pos, e1_pos, f1_pos, b1_pos)
    a2_pos, e2_pos, f2_pos, b2_pos = peak_trough_subdivide(
        der2, der2_peak, der2_trough)
    a2_pos, e2_pos, f2_pos, b2_pos = wash_peak_trough(
        a2_pos, e2_pos, f2_pos, b2_pos)

    if (len(a1_pos) == 0 or len(e1_pos) == 0 or len(f1_pos) == 0 or len(b1_pos) == 0 or len(a2_pos) == 0 or len(e2_pos) == 0 or len(f2_pos) == 0 or len(b2_pos) == 0):
        # print("here")
        return[], [], [], [], [], [], [], []

    a1 = np.array(der1[a1_pos])
    e1 = np.array(der1[e1_pos])
    f1 = np.array(der1[f1_pos])
    b1 = np.array(der1[b1_pos])
    # print("a1,e1,f1,b1:{}{}{}{}".format(a1.shape, e1.shape, f1.shape, b1.shape))

    a2 = np.array(der2[a2_pos])
    e2 = np.array(der2[e2_pos])
    f2 = np.array(der2[f2_pos])
    b2 = np.array(der2[b2_pos])
    # print("a2,e2,f2,b2:{}{}{}{}".format(a2.shape, e2.shape, f2.shape, b2.shape))

    ta1, tb1, te1, tf1 = get_t(a1_pos, e1_pos, f1_pos, b1_pos)
    ta2, tb2, te2, tf2 = get_t(a2_pos, e2_pos, f2_pos, b2_pos)
    # print("ta1,te1,tf1,tb1:{}{}{}{}".format(
    #     ta1.shape, te1.shape, tf1.shape, tb1.shape))
    # print("ta2,te2,tf2,tb2:{}{}{}{}".format(
    #     ta2.shape, te2.shape, tf2.shape, tb2.shape))

    # a = []
    # a.append(z/x)
    # a.append(e2/a2)
    # a.append((y-z)/x)
    # a.append(x)
    # a.append(t1/x)
    # a.append((tf1+t3)/t_pi)
    # a.append(s_base)
    # a.append(get_sVRI(data, peaks, troughs))
    # a.append(z)
    # a.append(ta2)

    # print(tf1.shape,t3.shape,t_pi.shape)

    return (z/x).reshape(-1, 1), (e2/a2).reshape(-1, 1), ((y-z)/x).reshape(-1, 1), (x).reshape(-1, 1), (t1/x).reshape(-1, 1), get_sVRI(data, peak_up, trough_down), (z).reshape(-1, 1), (ta2).reshape(-1, 1)


# 计算相邻波峰、波谷的横坐标差的平均数
def _pingjun(peaks, troughs):
    print((peaks[-1]-peaks[0])/(peaks.shape[0]-1))
    print((troughs[-1]-troughs[0])/(troughs.shape[0]-1))


if __name__ == '__main__':
    file_name = './data/1_200Hz_2019_3_5_10_9_11_rest1.txt'
    data = np.loadtxt(file_name, dtype=np.int16, skiprows=1)

    peaks, troughs = find_peak_trough(data, distance)

    f1, f2, f3, f4, f5, f8, f9, f10 = feature_extraction(
        data, peaks, troughs)
    print(f1.shape, f2.shape, f3.shape, f4.shape,
          f5.shape, f8.shape, f9.shape, f10.shape)
