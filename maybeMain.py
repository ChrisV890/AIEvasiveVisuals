import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import AdversarialPatch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data import random_split

#functions
def loader_to_numpy(loader):
    x_list = []
    y_list = []

    for images, labels in loader:
        x_list.append(images.numpy())   # already [0, 1] and NCHW from ToTensor()
        y_list.append(labels.numpy())

    x = np.concatenate(x_list, axis=0).astype(np.float32)
    y = np.concatenate(y_list, axis=0).astype(np.int64)
    return x, y


def to_one_hot(y, num_classes):
    return np.eye(num_classes)[y].astype(np.float32)



def evaluate_on_validation(classifier, val_loader):

    all_preds = []
    all_labels = []
    all_confidences = []

    for images, labels in val_loader:
        x = images.numpy().astype(np.float32)
        y = labels.numpy().astype(np.int64)

        preds = classifier.predict(x)
        probs = torch.softmax(torch.tensor(preds), dim=1).numpy()
        pred_classes = np.argmax(preds, axis=1)
        confidences = np.max(probs, axis=1)

        all_preds.append(pred_classes)
        all_labels.append(y)
        all_confidences.append(confidences)

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    all_confidences = np.concatenate(all_confidences)

    # Accuracy
    accuracy = np.mean(all_preds == all_labels)

    # Convert logits → probabilities (for confidence)
    probs = torch.softmax(torch.tensor(preds), dim=1).numpy()
    confidences = np.max(probs, axis=1)

    print("Validation Accuracy:", accuracy)
    print("Predicted class counts:", np.bincount(all_preds))
    print("True class counts:", np.bincount(all_labels))
    print("Sample confidences:", all_confidences[:10])

    return accuracy




#------------------------------------
#Device
#------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("GPU Primed\n")





#------------------------------------
#Model Creation
#------------------------------------

model = resnet18(weights=ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 2)  # binary classification

# optional: freeze early layers first
for param in model.layer1.parameters():
    param.requires_grad = False
for param in model.layer2.parameters():
    param.requires_grad = False

model = model.to(device)
print("ResNet Model Primed\n")


loss_CEL = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

classifier = PyTorchClassifier(
    model=model,
    loss=loss_CEL,
    optimizer=optimizer,
    input_shape=(3, 224, 224),
    nb_classes=2,
    clip_values=(-3, 3),
)
print("Classifier Primed\n")


#Normalize
weights = ResNet18_Weights.DEFAULT
preprocess = weights.transforms()

#image transofrmation and normalization
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),   # converts to [0 1]
    transforms.Normalize(mean = [0.485, 0.456, 0.406], std  = [0.229, 0.224, 0.225])
])

#Load Data
train_dataset = datasets.ImageFolder(
    root="AIPeopleDataset",   #EXTREMELY TEMPORARY I SWEAR
    transform=transform
)
print("Dataset Primed\n")
print("Class Mapping: ",train_dataset.class_to_idx)


#80% train, 20 validation
train_size = int(0.8 * len(train_dataset))
validation_size = len(train_dataset) - train_size

#split data
train_set, val_set = random_split(train_dataset, [train_size, validation_size])

#load data to go
train_loader = DataLoader(train_set, batch_size = 16, shuffle=True)
val_loader = DataLoader(val_set, batch_size = 16, shuffle=False)



#train classifier on data.
x_train, y_train = loader_to_numpy(train_loader)



classifier.fit(
    x_train,
    y_train,
    batch_size=16,
    nb_epochs=5,
    verbose=True
)


evaluate_on_validation(classifier, val_loader)
















