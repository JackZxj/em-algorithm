import os
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
from sklearn.datasets import make_moons
from sklearn.mixture import GaussianMixture as GMM
from sklearn.cluster import KMeans

base_path = os.path.dirname(os.path.realpath(__file__))
savePath = os.path.join(base_path, "images")
if not os.path.exists(savePath):
    os.makedirs(savePath)


def draw_ellipse(position, covariance, ax=None, **kwargs):
    """用给定的位置和协方差画一个椭圆"""
    ax = ax or plt.gca()

    # 将协方差转换为主轴
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)

    # 画出椭圆
    for nsig in range(1, 4):
        ax.add_patch(Ellipse(position, nsig * width, nsig * height,
                             angle, **kwargs))


def plot_gmm(gmm, X, label=True, ax=None, title=None):
    ax = ax or plt.gca()
    labels = gmm.fit(X).predict(X)
    if label:
        ax.scatter(X[:, 0], X[:, 1], c=labels, s=10, cmap='viridis', zorder=2)
    else:
        ax.scatter(X[:, 0], X[:, 1], s=10, zorder=2)
    # ax.axis('equal')

    if title:
        ax.set_title(title)

    w_factor = 0.2 / gmm.weights_.max()
    for pos, covar, w in zip(gmm.means_, gmm.covariances_, gmm.weights_):
        draw_ellipse(pos, covar, alpha=w * w_factor)

if __name__ == '__main__':
    Xmoon, ymoon = make_moons(200, noise=.1, random_state=0)
    plt.scatter(Xmoon[:, 0], Xmoon[:, 1])
    plt.gca().set_title('data set')
    plt.savefig(os.path.join(savePath, '1.png'))
    plt.cla()

    # 如果用GMM对数据拟合出两个成分，那么作为一个聚类模型的结果，效果将会很差
    gmm2 = GMM(n_components=2, covariance_type='full', random_state=0)
    plot_gmm(gmm2, Xmoon, title="2 classification with GMM")
    plt.savefig(os.path.join(savePath, '2.png'))
    plt.cla()

    #如果选用更多的成分而忽视标签，就可以找到一个更接近输入数据的拟合结果
    gmm16 = GMM(n_components=16, covariance_type='full', random_state=0)
    plot_gmm(gmm16, Xmoon, label=False, title="16 classification with GMM")
    plt.savefig(os.path.join(savePath, '3.png'))

    # 通过拟合后的GMM模型可以生成新的、与输入数据类似的随机分布函数
    Xnew, Ynew = gmm16.sample(400)
    plt.scatter(Xnew[:, 0], Xnew[:, 1], marker='*', c='red')
    plt.gca().set_title('generate similar data')
    plt.text(.99, .01, ('red : new'), transform=plt.gca().transAxes, size=10,
             horizontalalignment='right')
    plt.text(.99, .05, ('blue : old'), transform=plt.gca().transAxes, size=10,
             horizontalalignment='right')
    plt.savefig(os.path.join(savePath, '4.png'))
    plt.cla()

    plt.figure(figsize=(10, 5))
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.3, hspace=0.4)
    for index, k in enumerate((2, 4, 8, 16)):
        plt.subplot(2, 4, index+1)
        y_pred = KMeans(n_clusters=k, random_state=9).fit_predict(Xmoon)
        plt.scatter(Xmoon[:, 0], Xmoon[:, 1], c=y_pred, s=10)
        plt.gca().set_title('K-mean k=%d' %(k))

        plt.subplot(2, 4, index+5)
        gmm = GMM(n_components=k, covariance_type='full', random_state=0)
        plot_gmm(gmm, Xmoon, title='GMM k=%d' %(k)) 

    plt.savefig(os.path.join(savePath, '5.png'))
