import numpy as np
import pandas as pd


def sliding_window(data, sw_width=7, step=1):
    '''
    该函数实现窗口宽度为7、滑动步长为1的滑动窗口截取序列数据
    输入：data 原始数据 单变量(若要处理多变量可将train_seq = data[in_start:in_end]变为train_seq = data[in_start:in_end,:])
         sw_width 滑动窗口大小
         step 滑动步长
    '''
    X=[]
    in_start = 0
    for _ in range(len(data)):
        in_end = in_start + sw_width
        # 保证截取样本完整，最大元素索引不超过原序列索引，则截取数据；否则丢弃该样本
        if in_end < len(data):
            # 训练数据以滑动步长截取
            train_seq = data[in_start:in_end]
            X.append(train_seq)
        in_start += step
    return np.array(X)

def dataset_power(path='./pv.csv',sw_width=7, step=1,attribute='power'):
    '''
    该函数实现功率 power/发电量generation滑动截取并保存到当前目录
    输入：path 文件路径
         sw_width 窗口大小 默认为7
         step 滑动步长 默认为1
         attribute 属性名称 默认为功率
    返回：滑动后保存的数据
    '''
    ori_data = pd.read_csv(path)
    try:
        data = ori_data[attribute].values     #对应属性
    except:
        print('选择属性错误')
        return
    x = sliding_window(data,sw_width,step)
    #写入对应文件
    out = f'./{attribute}.txt'
    np.savetxt(out, x, fmt = '%s', delimiter = ',')
    return x

if __name__=='__main__':
    #滑动窗口为7，步长为1  功率属性
    dataset_power()
    #滑动窗口为7，步长为1  发电量属性
    dataset_power(attribute='generation')


