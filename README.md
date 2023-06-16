# FATE_HeteroLR_test
近似损失函数的一些问题测试

【1】近似损失函数对每个样本的损失最低点为ywx=2的位置，对标签为0损失最小的p为0.13左右，对标签为1损失最小的p为0.88左右
【2】当改变app_loss=X_b.T.dot(0.25*X_b.dot(theta) - 0.5*y_train) / len(y_train)至app_loss=X_b.T.dot(0.25*X_b.dot(theta) - 1*y_train) / len(y_train)，实际的log损失可以达到更低值，预测的概率值更低
【3】使用近似损失函数，onehot后由于特征稀疏，模型区分能力有明显下降，预测结果之间差异较小
【4】从实验9可以看到，若用近似梯度，同样学习率下，当特征较多时，onehot或者ordinal编码会导致WX的求和过大【尤其是ordinal】，产生梯度爆炸



【改进方式】
采用分段函数拟合损失函数，guest把与分段位置的差值取出设置为X_diff，匿名比较X_diff与host的wx值设置为X_host，若X_diff>X_host，则guest传分段左边的函数，否则则取右边的函数
