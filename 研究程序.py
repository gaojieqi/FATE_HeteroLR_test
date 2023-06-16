from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn import preprocessing
import numpy as np
from sklearn.datasets import make_classification
import seaborn as sns
sns.set(context='notebook',font='simhei',style='whitegrid')
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = ['SimHei']
plt.rcParams['axes.unicode_minus']=False
import warnings
warnings.filterwarnings('ignore')  # 不发出警告
from scipy.stats import norm

def sigmoid(z):
    return 1. / (1. + np.exp(-z))

def print_dist(data):
    plt.figure(figsize=(8,4))
    sns.distplot(data, bins=100, hist=True, kde=False, norm_hist=False,
            rug=True, vertical=False,label='distplot',
            axlabel='x轴',hist_kws={'color':'y','edgecolor':'k'},
            fit=norm)
    # 用标准正态分布拟合
    plt.legend()
    plt.grid(linestyle='--')
    plt.show()
#画出sigmoid函数图像
def ks_plot(predictions, labels, cut_point=100):
    good_len = len([x for x in labels if x == 0])  # 所有好客户数量
    bad_len = len([x for x in labels if x == 1])  # 所有坏客户数量
    predictions_labels = list(zip(predictions, labels))
    good_point = []
    bad_point = []
    diff_point = []  # 记录每个阈值点下的KS值

    x_axis_range = np.linspace(0, 1, cut_point)
    for i in x_axis_range:
        hit_data = [x[1] for x in predictions_labels if x[0] <= i]  # 选取当前阈值下的数据
        good_hit = len([x for x in hit_data if x == 0])  # 预测好客户数
        bad_hit = len([x for x in hit_data if x == 1])  # 预测坏客户数量
        good_rate = good_hit / good_len  # 预测好客户占比总好客户数
        bad_rate = bad_hit / bad_len  # 预测坏客户占比总坏客户数
        diff = good_rate - bad_rate  # KS值
        good_point.append(good_rate)
        bad_point.append(bad_rate)
        diff_point.append(diff)

    ks_value = max(diff_point)  # 获得最大KS值为KS值
    ks_x_axis = diff_point.index(ks_value)  # KS值下的阈值点索引
    ks_good_point, ks_bad_point = good_point[ks_x_axis], bad_point[ks_x_axis]  # 阈值下好坏客户在组内的占比
    threshold = x_axis_range[ks_x_axis]  # 阈值

    plt.plot(x_axis_range, good_point, color="green", label="好比率")
    plt.plot(x_axis_range, bad_point, color="red", label="坏比例")
    plt.plot(x_axis_range, diff_point, color="darkorange", alpha=0.5)
    plt.plot([threshold, threshold], [0, 1], linestyle="--", color="black", alpha=0.3, linewidth=2)

    plt.scatter([threshold], [ks_good_point], color="white", edgecolors="green", s=15)
    plt.scatter([threshold], [ks_bad_point], color="white", edgecolors="red", s=15)
    plt.scatter([threshold], [ks_value], color="white", edgecolors="darkorange", s=15)
    plt.title("KS={:.3f} threshold={:.3f}".format(ks_value, threshold))

    plt.text(threshold + 0.02, ks_good_point + 0.05, round(ks_good_point, 2))
    plt.text(threshold + 0.02, ks_bad_point + 0.05, round(ks_bad_point, 2))
    plt.text(threshold + 0.02, ks_value + 0.05, round(ks_value, 2))

    plt.legend(loc=4)
    plt.grid()
    plt.show()



###定义参数
n_samples=10000
n_features=5
n_informative=2
n_redundant=2
n_repeated=1
event_rate=0.03

bins_cnt=6
max_iter=1
verbose=1
learning_rate=0.01
class_weight=None #'balanced'

###定义建模方式
no_flag=1  #是否使用1/-1编码
bin_flag=1 #是否独热编码
make_data_flag=1 #是否手工创建数据集


if make_data_flag==0:
    ###自有数据集
    from sklearn.datasets import load_breast_cancer
    data=load_breast_cancer()
    x_data=data.data
    y_data=data.target
else:
    ###制作样本
    x_data, y_data = make_classification(n_samples=n_samples, n_features=n_features,
                                n_informative=n_informative, n_redundant=n_redundant,n_repeated=n_repeated,
                               weights=[1-event_rate,event_rate],
                                random_state=5829, shuffle=False,n_classes=2)

print('Event_rate:',y_data.sum()/len(y_data))

fea_cnt=x_data.shape[1]
scaler = preprocessing.StandardScaler().fit(x_data)
X_scaled_unbin = scaler.transform(x_data)

if bin_flag==0:
    #只做标准归一化
    X_scaled=X_scaled_unbin
else:
    #onehot编码
    bins=[bins_cnt for i in range(fea_cnt)]
    est = preprocessing.KBinsDiscretizer(encode='onehot',n_bins=bins,strategy='quantile').fit(X_scaled_unbin)
    X_scaled=est.transform(X_scaled_unbin)


x_train,x_test,y_train,y_test = train_test_split(X_scaled,y_data,stratify=y_data)

y_train_o=y_train.copy()


if no_flag:
    y_train[y_train==0]=-1

if bin_flag:
    X_b = x_train
    X_b_test = x_test
    theta = np.zeros(X_b.shape[1])   # 这个系数可以随机初始化，你可以初始化为正态分布的
else:
    X_b = np.hstack([np.ones((x_train.shape[0], 1)), x_train])
    X_b_test = np.hstack([np.ones((len(x_test), 1)), x_test])
    theta = np.zeros(X_b.shape[1]) #这个系数可以随机初始化，你可以初始化为正态分布的


def J(theta):
    y_hat = sigmoid(X_b.dot(theta))
    #0/1 loss
    z_o_loss=- np.sum(y_train_o*np.log(y_hat) + (1-y_train_o)*np.log(1-y_hat)) / len(y_train_o)
    no_o_loss = - np.sum( (y_train+1)/2* np.log(y_hat) + (1-y_train)/2 * np.log(1 - y_hat)) / len(y_train)
    #已加负号，不需要再加
    no_o_loss_app =  (np.sum(np.log(2)-0.5*X_b.dot(theta)*y_train+0.125*(X_b.dot(theta)*X_b.dot(theta)))) / len(y_train)
    return y_hat,no_o_loss,z_o_loss,no_o_loss_app

#梯度
def dJ(theta):
    x_sum=X_b.dot(theta)
    #0/1转化log梯度
    log_loss=X_b.T.dot(sigmoid(X_b.dot(theta)) - y_train) / len(y_train)

    #-1/1转化后log梯度
    ywx=x_sum*y_train
    d=(sigmoid(ywx)-1)*y_train
    log_loss_no=X_b.T.dot(d)/ len(y_train)
    #print(x_sum)
    #print(ywx)
    #print(d)
    #print(log_loss_no)

    #log_loss = .T.dot(X_b)
    #-1/1转化近似的log梯度
    app_loss=X_b.T.dot(0.25*X_b.dot(theta) - 0.5*y_train) / len(y_train)
    return x_sum,log_loss,log_loss_no,app_loss
    # return log_loss

#梯度下降法优化
apprloss_list=[]
iter_num = 0
max_iter = 300#
learing_rate = 0.1
while iter_num < max_iter:
    iter_num += 1
    last_theta = theta
    x_sum,logloss,log_loss_no,app_loss=dJ(theta)
    if no_flag:
        d=app_loss
    else :
        d=logloss
    theta = theta - learing_rate * d
    y_hat_last, J_theta_last,z_o_loss_last,no_o_loss_app_last=J(last_theta)
    y_hat,J_theta,z_o_loss,no_o_loss_app=J(theta)
    #if m#ax_iter<=5:
    #    print("thet#a:",theta)
    #    print("dj:", dj)
    apprloss_list.append(no_o_loss_app)
    if iter_num%10==0:
        print("【x_sum:", iter_num, "】", x_sum.mean())
        #print("【y_hat_last:", iter_num, "】", np.log(y_hat_last).max())
        #pr#int("【y_hat:",iter_num,"】",np.log(y_hat).max())
        #print("【der_log:", iter_num, "】", logloss)
        #print("【der_lno:",iter_num,"】",log_loss_no)
        #print("【der_app:", iter_num, "】", app_loss)
        #print("【Loss_last:",iter_num,"】",J_theta_last,"||",z_o_loss_last,"||",no_o_loss_app_last)
        print("【Loss:",iter_num,"】",J_theta,"||",z_o_loss,"||",no_o_loss_app)
    #if (abs(J_theta - J_theta_last) < 1e-4):
    #    break



y_predict = sigmoid(X_b_test.dot(theta))

print(y_predict[:10])

ks_plot(y_predict, y_test)

print_dist(y_predict)
print_dist(x_sum)
print_dist(apprloss_list)


