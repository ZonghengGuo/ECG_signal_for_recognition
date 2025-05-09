import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from dataloader import CustomDataset
from torchvision.models import resnet18
from models.distill import DistillableViT, DistillWrapper
from sklearn.metrics import confusion_matrix
import seaborn as sns
from torchvision import transforms


mode = 'Mel' # Todo: ['FFT', 'STFT', 'CWT', 'Mel']
database = 'vsmdb' # Todo: ['vsmdb', 'ecg_for_30', '6rp6', 'self_made']
window_length = 0.5 # Todo: [0.5, 1, 2]
# 定义设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


train_dataset = CustomDataset(f"databases/preprocess_spectrogram/preprocessing_{mode}/{database}/{window_length}", train=True, transform=transform)
test_dataset = CustomDataset(f"databases/preprocess_spectrogram/preprocessing_{mode}/{database}/{window_length}", train=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)

teacher = resnet18(pretrained = True)

if database == 'vsmdb' or database == "self_made":
    classes = 10
elif database == 'ecg_for_30':
    classes = 30
else:
    classes = 9

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

model.load_state_dict(torch.load(f"save_model/{mode}_{database}_{window_length}.pth"))
model.eval()

all_labels = []
all_probs = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        probabilities = torch.softmax(outputs, dim=1)  # 计算每个类别的概率
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probabilities.cpu().numpy())

conf_matrix = np.zeros((classes, classes))

for i in range(len(all_labels)):
    true_label = all_labels[i]
    predicted_probs = all_probs[i]
    for j in range(classes):
        conf_matrix[true_label, j] += predicted_probs[j]

conf_matrix_normalized = conf_matrix / conf_matrix.sum(axis=1, keepdims=True)
conf_matrix_percent = (conf_matrix_normalized * 100).round().astype(int)

window_length_str = str(window_length)[-1]
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix_percent, annot=True, cmap='Blues', fmt='d', xticklabels=range(classes), yticklabels=range(classes), annot_kws={"size": 12})
plt.xlabel('Predicted', fontsize=15)
plt.ylabel('True', fontsize=15)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.savefig(f"savefig/{mode}_{database}_{window_length_str}")
plt.show()


correct_preds = 0
total_samples = len(all_labels)

for i in range(total_samples):
    true_label = all_labels[i]
    predicted_probs = all_probs[i]
    correct_preds += predicted_probs[true_label]

accuracy = correct_preds / total_samples
print(f'Test Accuracy: {accuracy * 100:.2f}%')