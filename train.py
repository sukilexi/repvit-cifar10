import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from read_dataset import read_dataset
from repvit import repvit_m0_9

# set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# 读数据
batch_size = 128
train_loader, valid_loader, test_loader = read_dataset(batch_size=batch_size, pic_path='./data')
# 加载模型(使用预处理模型，修改最后一层，固定之前的权重)
n_class = 10
# model = models.resnet18()
model = repvit_m0_9()
model = model.to(device)
# 使用交叉熵损失函数
criterion = nn.CrossEntropyLoss().to(device)

# 开始训练
n_epochs = 250
valid_loss_min = np.Inf  # track change in validation loss
accuracy = []
lr = 1e-3
counter = 0
for epoch in tqdm(range(1, n_epochs + 1)):

    # keep track of training and validation loss
    train_loss = 0.0
    valid_loss = 0.0
    total_sample = 0
    right_sample = 0

    # 动态调整学习率
    if counter / 10 == 1:
        counter = 0
        lr = lr * 0.5
    optimizer = optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=1e-2)
    ###################
    # 训练集的模型 #
    ###################
    model.train()  # 作用是启用batch normalization和drop out
    for data, target in train_loader:
        data = data.to(device)
        target = target.to(device)
        # clear the gradients of all optimized variables（清除梯度）
        optimizer.zero_grad()
        # forward pass: compute predicted outputs by passing inputs to the model
        # (正向传递：通过向模型传递输入来计算预测输出)
        output = model(data).to(device)  # （等价于output = model.forward(data).to(device) ）
        # calculate the batch loss（计算损失值）
        loss = criterion(output, target)
        # backward pass: compute gradient of the loss with respect to model parameters
        # （反向传递：计算损失相对于模型参数的梯度）
        loss.backward()
        # perform a single optimization step (parameter update)
        # 执行单个优化步骤（参数更新）
        optimizer.step()
        # update training loss（更新损失）
        train_loss += loss.item() * data.size(0)

    ######################
    # 验证集的模型#
    ######################

    model.eval()  # 验证模型
    for data, target in valid_loader:
        data = data.to(device)
        target = target.to(device)
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data).to(device)
        # calculate the batch loss
        loss = criterion(output, target)
        # update average validation loss
        valid_loss += loss.item() * data.size(0)
        # convert output probabilities to predicted class(将输出概率转换为预测类)
        _, pred = torch.max(output, 1)
        # compare predictions to true label(将预测与真实标签进行比较)
        correct_tensor = pred.eq(target.data.view_as(pred))
        # correct = np.squeeze(correct_tensor.to(device).numpy())
        total_sample += batch_size
        for i in correct_tensor:
            if i:
                right_sample += 1
    print("Accuracy:", 100 * right_sample / total_sample, "%")
    accuracy.append(right_sample / total_sample)

    # 计算平均损失
    train_loss = train_loss / len(train_loader.sampler)
    valid_loss = valid_loss / len(valid_loader.sampler)

    # 显示训练集与验证集的损失函数
    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
        epoch, train_loss, valid_loss))

    # 如果验证集损失函数减少，就保存模型。
    if valid_loss <= valid_loss_min:
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min, valid_loss))
        torch.save(model.state_dict(), 'checkpoint/repvit_m0_9_new_cifar10.pt')
        valid_loss_min = valid_loss
        counter = 0
    else:
        counter += 1