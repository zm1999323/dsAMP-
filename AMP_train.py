import os
import sys
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from setnet2 import Classifier
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score
from utils import ProteinDataset, h5py_to_tensor
from sklearn.model_selection import StratifiedKFold


torch.manual_seed(1)



def train_and_validate_model(train_loader, test_loader, net, loss_function, optimizer, epochs):
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    precisions = []
    recalls = []
    f1_scores = []

    best_val_accuracy = 0.0  # 用于保存最佳验证准确度
    best_model_state = None  # 用于保存最佳模型参数

    for epoch in range(epochs):
        net.train()
        running_loss = 0.0
        correct = 0
        total = 0
        train_bar = tqdm(train_loader, file=sys.stdout)

        for data in train_bar:
            embeddings, labels = data
            optimizer.zero_grad()
            outputs = net(embeddings.to(device))
            loss = loss_function(outputs, labels.to(device))
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            predict_y = torch.max(outputs, dim=1)[1]
            total += labels.size(0)
            correct += torch.eq(predict_y, labels.to(device)).sum().item()

            train_bar.desc = f"Train Epoch [{epoch + 1}/{epochs}] Loss: {loss:.3f}"

        avg_loss = running_loss / len(train_loader)
        avg_accuracy = 100 * correct / total

        train_losses.append(avg_loss)
        train_accuracies.append(avg_accuracy)

        net.eval()
        val_correct = 0
        val_total = 0
        predictions = []
        true_labels = []
        val_running_loss = 0.0

        with torch.no_grad():
            val_bar = tqdm(test_loader, file=sys.stdout)
            for val_data in val_bar:
                val_embeddings, val_labels = val_data
                outputs = net(val_embeddings.to(device))
                val_loss = loss_function(outputs, val_labels.to(device))
                val_running_loss += val_loss.item()
                predict_y = torch.max(outputs, dim=1)[1]
                val_total += val_labels.size(0)
                val_correct += torch.eq(predict_y, val_labels.to(device)).sum().item()

                predictions.extend(predict_y.cpu().numpy())
                true_labels.extend(val_labels.cpu().numpy())

        avg_val_loss = val_running_loss / len(test_loader)
        val_accuracy = 100 * val_correct / val_total
        precision = precision_score(true_labels, predictions)
        recall = recall_score(true_labels, predictions)
        f1 = f1_score(true_labels, predictions)

        val_losses.append(avg_val_loss)
        val_accuracies.append(val_accuracy)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
        print(f"\nEpoch [{epoch + 1}/{epochs}]")
        print(f"Train Loss: {avg_loss:.4f}, Train Accuracy: {avg_accuracy:.2f}%")
        print(
            f"Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}")

        # 保存在验证集上表现最好的模型
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model_state = net.state_dict()

    return train_losses, train_accuracies, val_losses, val_accuracies, precisions, recalls, f1_scores, best_model_state


def create_protein_datasets(train_pos, train_neg):
    X_train_positive = h5py_to_tensor(h5_path=train_pos)
    X_train_negative = h5py_to_tensor(h5_path=train_neg)
    # X_test_positive = h5py_to_tensor(h5_path=test_pos)
    # X_test_negative = h5py_to_tensor(h5_path=test_neg)

    # 将正类和负类的嵌入向量合并
    X_train_combined = np.concatenate((X_train_positive, X_train_negative), axis=0)
    # X_test_combined = (np.concatenate((X_test_positive, X_test_negative), axis=0))

    # 创建对应的标签列表
    y_train_combined = [1] * len(X_train_positive) + [0] * len(X_train_negative)
    # y_test_combined = [1] * len(X_test_positive) + [0] * len(X_test_negative)

    # 创建数据集对象
    train_dataset = ProteinDataset(X_train_combined, y_train_combined)
    # test_dataset = ProteinDataset(X_test_combined, y_test_combined)

    return train_dataset


train_pos = '/home/wuyou/project/pythonProject5/pythonProject/pythonProject1/anti-bacterial/dataset/anti_bacterial_pos_cdhit90_train80.h5'
train_neg = '/home/wuyou/project/pythonProject5/pythonProject/pythonProject1/anti-bacterial/dataset/anti_bacterial_neg_cdhit90_train80.h5'

train_dataset = create_protein_datasets(train_pos, train_neg)

# 定义超参数和设备
batch_size = 128
epochs = 60
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("using {} device.".format(device))

# 定义十折交叉验证
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# 初始化列表用于保存所有折叠的指标和最佳模型参数
all_train_losses = []
all_train_accuracies = []
all_val_losses = []
all_val_accuracies = []
all_precisions = []
all_recalls = []
all_f1_scores = []
all_best_model_states = []

for fold, (train_index, test_index) in enumerate(skf.split(train_dataset, train_dataset.get_targets())):
    print(f"\nFold {fold + 1}/{skf.n_splits}")

    # 根据索引创建对应的子集和 DataLoader
    train_subset = torch.utils.data.Subset(train_dataset, train_index)
    test_subset = torch.utils.data.Subset(train_dataset, test_index)

    train_loader = torch.utils.data.DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_subset, batch_size=batch_size, shuffle=True)

    # 创建模型、损失函数和优化器
    num_classes = 2

    net = Classifier(num_classes=num_classes)
    net_weights_path = '/home/wuyou/project/pythonProject5/pythonProject/pythonProject1/AMP/model3/best_model_fold_3.pth'
    net.load_state_dict(torch.load(net_weights_path))

    net.to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.00001, weight_decay=0.0030)

    # 训练和验证模型
    train_losses, train_accuracies, val_losses, val_accuracies, precisions, recalls, f1_scores, best_model_state = train_and_validate_model(train_loader, test_loader, net, loss_function, optimizer, epochs)

    # 保存在验证集上表现最好的模型参数
    all_best_model_states.append(best_model_state)


    # 将折叠的指标添加到列表中
    all_train_losses.append(np.array(train_losses))
    all_train_accuracies.append(np.array(train_accuracies))
    all_val_losses.append(np.array(val_losses))
    all_val_accuracies.append(np.array(val_accuracies))
    all_precisions.append(precisions)
    all_recalls.append(recalls)
    all_f1_scores.append(f1_scores)

# 保存最佳模型参数
for i, best_model_state in enumerate(all_best_model_states):
    torch.save(best_model_state, f"best_model_fold_{i + 1}.pth")

avg_train_losses = np.mean(all_train_losses, axis =0)
avg_val_losses = np.mean(all_val_losses,axis=0)
avg_train_accuracies = np.mean(all_train_accuracies,axis=0)
avg_val_accuracies = np.mean(all_val_accuracies,axis=0)

# 绘制整体的训练和验证指标

plt.figure(figsize=(8, 5))

plt.subplot(1, 2, 1)
plt.plot(range(1, epochs + 1), avg_train_losses, label='Train Loss')
plt.plot(range(1, epochs + 1), avg_val_losses, label='Val Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss and Val Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(1, epochs + 1), avg_train_accuracies, label='Train Accuracy')
plt.plot(range(1, epochs + 1), avg_val_accuracies, label='Val Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()


plt.tight_layout()
plt.savefig('average_antibacterial.png')



