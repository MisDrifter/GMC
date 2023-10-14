import numpy as np
import matplotlib.pyplot as plt
def eval(G, s, fx, x, y, h):
    S = []
    n = x.shape[0]
    for g in G:
        S.append(g(x)@s(fx,x,y,h)/n)
    return np.array(S)

def eval2(G, s, fx, x, y, h):
    S = []
    for g in G:
        n = g(x).sum()
        S.append(g(x)@s(fx,x,y,h)/n)
    return np.array(S)

def plot_result(category,result1, result2, x_label_name, y_lable_name, title, base=None):
    # 创建一个包含两个子图的图表
    fig, ax = plt.subplots(figsize=(len(category)*2,5))

    # 设置柱状图的位置
    x1 = np.arange(len(category))

    # 绘制第一个柱状图
    ax.bar(x1 - 0.2, result1, width=0.4, label='GMC')

    # 绘制第二个柱状图
    ax.bar(x1 + 0.2, result2, width=0.4, label='baseline')

    # 绘制横线（示例：在y=20的位置绘制横线）
    if not (base is None):
        ax.axhline(y=base, color='red', linestyle='--', label='alpha')

    # 添加标签和标题
    ax.set_xlabel(x_label_name, fontsize=18)
    ax.set_ylabel(y_lable_name, fontsize=18)
    ax.set_title(title, fontsize=18)

    # 添加图例
    ax.legend(fontsize=15)

    # 设置x轴刻度标签
    ax.set_xticks(x1)
    ax.set_xticklabels(category, fontsize=15)

    # 显示图表
    plt.show()