# coding=utf-8
import random
import argparse
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def init_points(class_num, x):
    return random.sample(x, class_num)


def divide_by_center_pnts(center_pnts, pnts):
    belongs = []
    for pnt in pnts:
        # 该点到各个类中心的距离
        distances = []
        for center_pnt in center_pnts:
            each_feature_distance = list(map(lambda p_cp: math.pow(p_cp[0] - p_cp[1], 2), zip(pnt, center_pnt)))
            distance = math.sqrt(sum(each_feature_distance))
            distances.append(distance)
        # 得到该点属于某个类别
        belong = distances.index(min(distances))
        belongs.append(belong)
    return belongs


def move_center_pnts(belongs, pnts):
    feature_num = len(pnts[0])
    class_num = max(belongs) + 1
    pnts_num = len(belongs)
    center_pnts = [[0] * feature_num] * class_num
    # 计算各个类的各个（特征之和）
    for index in range(pnts_num):
        center_pnt = center_pnts[belongs[index]]
        pnt = pnts[index]
        center_pnts[belongs[index]] = list(map(lambda x_y: x_y[0] + x_y[1], zip(center_pnt, pnt)))
    # 各个类的各个（特征之和）除以包含的点个个数。求取均值
    for c_index in range(class_num):
        belong_pnt_num = belongs.count(c_index)
        for f_index in range(feature_num):
            center_pnts[c_index][f_index] /= belong_pnt_num if belong_pnt_num != 0 else 1
    return center_pnts


def cal_losses_sum(center_pnts, belongs, pnts):
    losses_sum = 0.
    feature_num = len(pnts[0])
    pnts_num = len(pnts)
    for index in range(pnts_num):
        center_pnt = center_pnts[belongs[index]]
        pnt = pnts[index]
        loss = 0.
        # 计算当前点到所属类的中心点的距离
        for f_index in range(feature_num):
            loss += math.pow(center_pnt[f_index] - pnt[f_index], 2)
        losses_sum += math.sqrt(loss)
    return losses_sum


def k_means(class_num, pnts):
    # 初始化class_num个中心点
    center_pnts = init_points(class_num, pnts)
    # 记录上一次分类情况的损失函数，用于判断是否需要继续
    pre_loss_sum = 1e10
    while True:
        # 根据中心点对所有点进行聚类, 得到每个点的类别及其对该类的距离贡献
        belongs = divide_by_center_pnts(center_pnts, pnts)

        # 每个中心点进行偏移，移动到所有属于该类的平均值中心处
        center_pnts = move_center_pnts(belongs, pnts)

        # 计算当前分类的损失值——各个类的欧拉距离(误差)之和
        cur_loss_sum = cal_losses_sum(center_pnts, belongs, pnts)

        # 如果该次改变对距离(误差)有优化，则继续循环，否则退出结束
        if cur_loss_sum < pre_loss_sum:
            pre_loss_sum = cur_loss_sum
        else:
            break

    return belongs, pre_loss_sum, center_pnts


def show(x, y):
    figure, ax = plt.subplots()
    plt.title("CLY_K-MEANS")
    ax.add_line(Line2D(x, y, linewidth=1, color='blue'))
    ax.set_xlim(left=0, right=len(x)+1)
    ax.set_xlabel('CLASS_NUM')
    ax.set_ylim(bottom=0, top=max(y)*1.1)
    ax.set_ylabel('LOSS')
    for a, b in zip(x, y):
        plt.text(a, b, round(b, 2), ha='center', va='bottom', fontsize=9)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('f',type=str, help='file(.npy) path')
    parser.add_argument('-c',type=int, help='class num')
    args = parser.parse_args()
    if args.f:
        PNTS_PATH = args.f
        pnts = np.load('sample_pnts.npy').tolist()
    if args.c:
        CLASS_NUM = args.c
        belongs, loss, center_pnts = k_means(class_num=CLASS_NUM, pnts=pnts)
        print('each point(1,2,3...) belong to:')
        print(belongs)
        print('the loss of such divide:')
        print(loss)
        print('the classes(1,2,3...)\'s center points:')
        print(center_pnts)
    else:
        # calculate each loss of given CLASS_NUM(0 - len(pnts)) for you can find the best value of CLASS_NUM
        # # 由于K-MEANS的随机初始化的影响，所以多次测量取最小值
        check_time = 15
        x = [i for i in range(1, len(pnts))]
        y = []
        for num in range(1, len(pnts)):
            each_loss = []
            for i in range(check_time):
                _, loss, _ = k_means(num, pnts)
                each_loss.append(loss)
            y.append(min(each_loss))
        show(x, y)