import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import AdversarialPatch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def preprocess(images):
    x = np.array(images)
    x = x.transpose(0, 3, 1, 2)  # NHWC → NCHW
    x = x / 255.0
    return x.astype(np.float32)


#-------
#Create Model
#-------

model = resnet18(weights=ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 2)  # binary classification

#loss_fn = nn.CrossEntropyLoss()
#optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

classifier = PyTorchClassifier(
    model=model,
    loss=nn.CrossEntropyLoss(),
    optimizer=torch.optim.Adam(model.parameters(), lr=1e-4),
    input_shape=(3, 224, 224),
    nb_classes=2,
    clip_values=(0, 1),
)






#visual pattern generation
attack = AdversarialPatch(
    classifier,
    patch_shape=(3, 50, 50),  # <-- REQUIRED
    rotation_max=22.5,
    scale_min=0.5,
    scale_max=1.0,
    learning_rate=5.0,
    max_iter=100,
    batch_size=8,
)





#DATASET STUFF
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),   # converts to [0,1]
])

train_dataset = datasets.ImageFolder(
    root="testtpeeple/Human Faces Dataset",
    transform=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True
)





#data to numpy for ART
x_batch, y_batch = next(iter(train_loader))

x = x_batch.numpy().astype(np.float32)   # NO transpose
y = y_batch.numpy()

y_onehot = np.eye(2)[y]

print(x.shape)

patch, _ = attack.generate(x=x, y=y_onehot)

