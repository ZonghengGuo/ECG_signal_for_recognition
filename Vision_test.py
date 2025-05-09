import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from dataloader import CustomDataset
from torchvision.models import resnet18
from models.distill import DistillableViT, DistillWrapper
from torchvision import transforms
from sklearn.manifold import TSNE

mode = 'Mel' # Todo: ['FFT', 'STFT', 'CWT', 'Mel']
database = 'vsmdb' # Todo: ['vsmdb', 'ecg_for_30', '6rp6', 'self_made']
window_length = 0.8 # Todo: [0.5, 1, 2]
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

correct_preds = 0
total_samples = len(all_labels)

for i in range(total_samples):
    true_label = all_labels[i]
    predicted_probs = all_probs[i]
    correct_preds += predicted_probs[true_label]

accuracy = correct_preds / total_samples
print(f'Test Accuracy: {accuracy * 100:.2f}%')

all_labels = np.array(all_labels)
all_probs = np.array(all_probs)

tsne = TSNE(n_components=2, random_state=42)
probs_2d = tsne.fit_transform(all_probs)

window_length_str = str(window_length)[-1]

plt.figure(figsize=(10, 8))
scatter = plt.scatter(probs_2d[:, 0], probs_2d[:, 1], c=all_labels, cmap='viridis', alpha=0.7, s=500)
plt.colorbar(scatter, label='Classes')
plt.title('t-SNE Visualization of Classification Results')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.savefig(f"save_feature_space/{mode}_{database}_{window_length_str}")
plt.show()