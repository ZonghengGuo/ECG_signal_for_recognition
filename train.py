import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
from dataloader import CustomDataset
from tqdm import tqdm
from torchvision.models import resnet18
from vit_pytorch.t2t import T2TViT
from models.distill import DistillableViT, DistillWrapper


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

mode = 'Mel' # Todo: ['FFT', 'STFT', 'CWT', 'Mel']
database = 'vsmdb' # Todo: ['vsmdb', 'ecg_for_30', '6rp6', 'self_made']
window_length = 0.8 # Todo: [0.5, 1, 2]

train_dataset = CustomDataset(f"databases/preprocess_spectrogram/preprocessing_{mode}/{database}/{window_length}", train=True, transform=transform)
test_dataset = CustomDataset(f"databases/preprocess_spectrogram/preprocessing_{mode}/{database}/{window_length}", train=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)
print("training dataset length:", len(train_dataset))
print("testing dataset length:", len(test_dataset))

if database == 'vsmdb' or database == 'self_made':
    classes = 10
elif database == 'ecg_for_30':
    classes = 30
else:
    classes = 9

teacher = resnet18(pretrained = True)

model = DistillableViT(
    image_size = 64,
    patch_size = 8,
    num_classes = classes,
    dim = 128,
    depth = 2,
    heads = 8,
    mlp_dim = 128,
    dropout = 0.2,
    emb_dropout = 0.2
)

distiller_loss = DistillWrapper(
    student = model,
    teacher = teacher,
    temperature = 3,           # temperature of distillation
    alpha = 0.5,               # trade between main loss and distillation loss
    hard = False               # whether to use soft or hard distillation
)


model = model.to(device)
teacher = teacher.to(device)

criterion = distiller_loss.cuda()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train(train_loader, model, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(data, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

    avg_loss = total_loss / len(train_loader)
    accuracy = correct / len(train_loader.dataset)
    return avg_loss, accuracy


def test(test_loader, model, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # print(output)
            total_loss += criterion(data, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    avg_loss = total_loss / len(test_loader)
    accuracy = correct / len(test_loader.dataset)
    return avg_loss, accuracy


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

if database == '6rp6':
    num_epochs = 300
else:
    num_epochs = 200

best_accuracy = 0.0

train_acc = []
test_acc = []
last_acc = []

y_true = []
y_pred = []

for epoch in tqdm(range(num_epochs)):
    train_loss, train_accuracy = train(train_loader, model, criterion, optimizer, device)
    test_loss, test_accuracy = test(test_loader, model, criterion, device)
    train_acc.append(train_accuracy)
    test_acc.append(test_accuracy)
    print(
        f'Epoch {epoch}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Train Accuracy: {train_accuracy*100:.2f}%, Test Accuracy: {test_accuracy*100:.2f}%')

    if test_accuracy > best_accuracy:
        best_accuracy = test_accuracy
        torch.save(model.state_dict(), f'save_model/{mode}_{database}_{window_length}.pth')
        # print(f"Saved new best model with accuracy: {best_accuracy*100:.2f}%")
    if epoch > num_epochs - 10:
        last_acc.append(test_accuracy)

print("最后十轮测试结果平均值", np.mean(last_acc))

window_length_str = str(window_length)[-1]

plt.figure(figsize=(10, 5))
plt.grid()
plt.plot(train_acc, label='Train Acc')
plt.plot(test_acc, label='Test Acc')
plt.title('Train vs Test Acc')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig(f'Output/Vit+{mode}+{database}+{window_length_str}')
plt.show()
