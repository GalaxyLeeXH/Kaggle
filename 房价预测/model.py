import pandas as pd
import torch
import matplotlib
import matplotlib.pyplot as plt
import numpy
import pandas
import os
import seaborn as sns


device = torch.device("cuda" if torch.cuda.is_available() else "CPU")
print("this is device: ", device, end='\n\n')

print("=========================读入数据=========================")
# 读入train和test的数据
train_data = pd.read_csv("DataSet/train.csv")
test_data = pd.read_csv("DataSet/test.csv")
print("train_data.shape: ", train_data.shape)
print("test_data.shape: ", test_data.shape)

# 查看train和test差异的列名，可以看到test.csv文件里没有SalePrice这一项，这是为了防止模型记住房价
# 所以模型训练的根本就是通过属性去拟合房价
different_columns = set(train_data.columns).symmetric_difference(set(test_data.columns))
print("train和test的列差： ",different_columns)

# 把train和test的数据进行融合之后处理非数值的值
# train_data.iloc[:, 1:-1]选中train_data的所有行，以及第二列到倒数第二列，切片操作前闭后开
# 跳过id和房价两个属性，id是避免模型记住房价，跳过房价是因为test没有房价信息
all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))
print("all_features.shape: ", all_features.shape, end='\n\n')

print("==============================处理缺省值和属性==================================")

# 获取全是数字的属性的名字
numeric_features = all_features.dtypes[all_features.dtypes != "object"].index
print("numeric_features: ", numeric_features, end='\n\n')

# 对于数字为0的位置做标准化处理，对应位置赋值，保证数据是在同一分布上，确保能有个不错的收敛速度，避免偏差
all_features[numeric_features] = all_features[numeric_features].apply(lambda x: (x - x.mean()) / (x.std()))
print("标准化处理过后的特征文件：\n", all_features.head(), end='\n\n')

# 经过标准化处理过后，把所有数值属性的缺失值置为0,可视化缺失值的位置，使用seaborn的热度图
fig, ax = plt.subplots(1, 4, figsize=(12, 8))
# 第一个子图，有空数值的情况
missing_rows = all_features[all_features.isna().any(axis=1)]
sns.heatmap(all_features[numeric_features].isna(), cbar=False, ax=ax[0])
ax[0].set_title("Contain Numeric Null")
# 第二个子图，填充缺失值之后的情况
all_features[numeric_features] = all_features[numeric_features].fillna(0)
sns.heatmap(all_features[numeric_features].isna(), cbar=False, ax=ax[1])
ax[1].set_title("Fixed Numeric Null")



# 数值的空缺值处理完之后处理缺失值
# 还未处理的包含Null的数据集
sns.heatmap(all_features.isna(), cbar=False, ax=ax[2])
ax[2].set_title("Contain attr Null")
# 处理过后把所有Null值填满的数据集
# dummy_na = True为每个缺失值NA创建一个独立的指示符dummy变量，有缺失就会生成一个列，用来表征该特征是否为缺失值
# 当使用dummy_na之后，Na就会被新的指示列表示，isna就不会再把它当成na检测，所以na全部被覆盖了
all_features = pd.get_dummies(all_features, dummy_na=True)
sns.heatmap(all_features.isna(), cbar=False, ax=ax[3])
ax[3].set_title("Fixed attr Null")
print("all_features.shape: ", all_features.shape, end='\n\n')
# pandas切片必须用iloc，否则切不了不可以直接像切列表一样切
print("all_features中的nan列: \n", [col for col in all_features.columns if col.endswith('_nan')], end='\n\n')
plt.tight_layout()
plt.show()


print("===============================拆分训练集和测试集=====================================")
# 训练集数据条数
n_train = train_data.shape[0]
# n条以前做训练数据、
print("all_features.dtype: ", all_features.dtypes, end='\n\n')
all_features = all_features.astype(float)
print("after change all_features.dtype: ", all_features.dtypes, end='\n\n')
train_features = torch.tensor(all_features[: n_train].values, dtype=torch.float32)
# n条以后做测试数据
test_features = torch.tensor(all_features[n_train:].values, dtype=torch.float32)
# train_labels - 特征对应的房价，reshape(-1, 1)只要一列，转换成为二维数组，多少行不管但是只要一列，去对应特征值
train_labels = torch.tensor(train_data.SalePrice.values.reshape(-1, 1), dtype=torch.float32)
print("test_features: \n", test_features, end='\n\n')
print("train_features.shape: ", train_features.shape)
print("test_features.shape: ", test_features.shape)
print("train_labels: ", train_labels.shape, end='\n\n')


print("===================================数据分批======================================")
batch_size = 32
# 将每个样本的特征和标签组合成为一个元组，是一种数据包装器，包装起来和数据加载器DataLoader结合使用
dataset = torch.utils.data.TensorDataset(train_features, train_labels)
train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
print(f'每一批{len(next(iter(train_loader))[0])}个，一共{len(train_loader)}批数据', end='\n\n')
for batch_idx, (features, labels) in enumerate(train_loader):
    print("batch_id: \n", batch_idx, '\n', "features: \n", features, '\n', "salary labels: \n", labels)

print("============================定义网络=======================================")
class Mynet(torch.nn.Module):
    def __init__(self, in_put, hidden, hidden1, out_put):
        super().__init__()
        # 定义一个三层的神经网络 - 两个全连接层和一个输出层
        # 这一块负责定义网络架构
        self.linear1 = torch.nn.Linear(in_put, hidden)
        self.linear2 = torch.nn.Linear(hidden, hidden1)
        self.linear3 = torch.nn.Linear(hidden1, out_put)

    def forward(self, data):
        # 这一块负责具体的传输
        x = self.linear1(data)
        x = torch.relu(x)
        x = self.linear2(x)
        x = torch.relu(x)
        x = self.linear3(x)
        return x

# 输入的特征个数
in_features = train_features.shape[1]
# 神经元个数、输出个数
hidden, hidden1, out_put = 200, 200, 1
# 如果存在cuda则使用gpu训练网络模型
model = Mynet(in_features, hidden, hidden1, out_put).to(device)

# loss function
loss = torch.nn.MSELoss()

# 梯度下降法
learn_rate = 1e-2  # lr = 0.01
optimizer = torch.optim.Adam(model.parameters(), learn_rate)
print("in_features: ", in_features, end='\n\n')
print("train_features.shape: ", test_features.shape, end='\n\n')
print(model, end='\n\n')


print("==============================================训练神经网络==========================================")
# 设置训练轮次为200次
epochs = 200
def train(train_loader):
    # 存储每个epoch的损失值
    train_ls = []
    for epoch in range(epochs):
        # 记录一个训练周期的总损失
        loss_sum = 0
        # 嵌套循环便利train_loader中的每个批次，其中训练数据和标签相对应
        for train_batch, labels_batch in train_loader:
            train_batch, labels_batch = train_batch.to(device), labels_batch.to(device)
            # model(train_batch)是模型对输入数据的预测，然后和labels_batch做均方损失
            l = loss(model(train_batch), labels_batch)
            # 在进行反向传播之前清除模型参数的梯度
            optimizer.zero_grad()
            # 执行反向传播，计算损失函数关于模型参数的梯度
            l.backward()
            # 更新模型参数，负责根据计算出的梯度调整模型的参数
            optimizer.step()
            # l.item()将损失值从张量转换为普通数字
            print("每次迭代的损失值：", l.item())
            loss_sum += l.item()
        # 把每次的损失值记录下来，做一个可视化
        train_ls.append(loss_sum)
    print("train_ls: ", train_ls)
    plt.plot(range(epochs), train_ls)
    plt.show()
train(train_loader)

print("=====================================测试神经网络=========================================")
def test(test_features):
    # test_features是一个二维数组，每处理一行都会在preds中生成一行，所以preds是二维数组
    test_features = test_features.to(device)
    # detach(): 从计算图中分离预测结果，不会在后续操作中计算梯度
    # to('cpu'): 预测结果移回CPU，因为不需要使用GPU加速
    # .numpy(): 将预测结果从Pytorch张量转换为Numpy数组
    preds = model(test_features).detach().to("cpu").numpy()
    # squeeze()方法去除结果中的单维度条目，把[n, 1]转换为[n]
    # 由于preds一个一个生成以后是二维数组，所以squeeze一下变成一维数组
    print("preds: \n", preds)
    print("preds after squeeze: \n", preds.squeeze(), end='\n\n')
    print("preds shape: ", preds.shape, end='\n\n')
    print("preds after squeeze's shape: ", preds.squeeze().shape)
    # 将预测结果squeeze之后作为一个新的列添加到DataFrame中
    test_data['SalePrice'] = pandas.Series(preds.squeeze())
    return pandas.concat([test_data['Id'], test_data['SalePrice']], axis=1)

submission = test(test_features)
print("submission: \n", submission)
# index = False - 创建csv文件时不添加索引
submission.to_csv("DataSet/submission.csv", index=False)