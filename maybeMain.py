import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import AdversarialPatch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from attackUtility import generate_adversarial_patch,softmax_np, loader_to_numpy, show_clean_vs_patched, show_least_confident_patched_with_clean


#------------------------------------
#Device
#------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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



#------------------------------------
#Load data, split
#------------------------------------

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
print("Class Mapping: ",train_dataset.class_to_idx)


#80% train, 20 validation
train_size = int(0.8 * len(train_dataset))
validation_size = len(train_dataset) - train_size

#split data
train_set, val_set = random_split(train_dataset, [train_size, validation_size])

#load data to go
train_loader = DataLoader(train_set, batch_size = 16, shuffle=True)
val_loader = DataLoader(val_set, batch_size = 16, shuffle=False)
x_train, y_train = loader_to_numpy(train_loader)
x_val, y_val = loader_to_numpy(val_loader)






#------------------------------------
#Train classification model on data
#------------------------------------


#train classifier
classifier.fit(
    x_train,
    y_train,
    batch_size=16,
    nb_epochs=5,
    verbose=True
)



#----------------------------------------------------
#Create an adversarial patch and train it on dataset
#----------------------------------------------------


# Create attack
attack = AdversarialPatch(
    classifier=classifier,
    patch_shape=(3, 75, 75),
    rotation_max=22.5,
    scale_min=0.7,
    scale_max=1.0,
    learning_rate=5.0,
    max_iter=200,
    batch_size=8,
)

# Generate patch
patch, mask, x_batch, y_batch = generate_adversarial_patch(
    attack,
    train_loader,
    num_classes=2
)



# ------------------------------------
# Evaluate the SAME validation batch
# clean first, then patched
# ------------------------------------
x_eval = x_val
y_eval = y_val



# Clean predictions
clean_logits = classifier.predict(x_eval)
clean_probs = softmax_np(clean_logits)
clean_pred_classes = np.argmax(clean_probs, axis=1)
clean_confidences = np.max(clean_probs, axis=1)
clean_accuracy = np.mean(clean_pred_classes == y_eval)

# Patched predictions on the SAME images
x_eval_patched = attack.apply_patch(
    x_eval,
    scale=0.4,
    patch_external=patch
)

patched_logits = classifier.predict(x_eval_patched)
patched_probs = softmax_np(patched_logits)
patched_pred_classes = np.argmax(patched_probs, axis=1)
patched_confidences = np.max(patched_probs, axis=1)
patched_accuracy = np.mean(patched_pred_classes == y_eval)

# Metrics
print("\n--- Same Batch Comparison ---")
print("Clean Accuracy:", clean_accuracy)
print("Patched Accuracy:", patched_accuracy)
print("Accuracy Drop:", clean_accuracy - patched_accuracy)
print("Mean Clean Confidence:", np.mean(clean_confidences))
print("Mean Patched Confidence:", np.mean(patched_confidences))
print("Mean Confidence Drop:", np.mean(clean_confidences) - np.mean(patched_confidences))
print("Attack Success Rate:", np.mean((clean_pred_classes == y_eval) & (patched_pred_classes != y_eval)))






#-----------------------------------------------
#good ole visualizations, still needs some work
#-----------------------------------------------

#Shows pictuas
show_clean_vs_patched(
    x=x_eval,
    y=y_eval,
    attack=attack,
    patch=patch,
    classifier=classifier,
    n=10,
    scale=0.4
)


#shows least confident patched pics
show_least_confident_patched_with_clean(
    x=x_val,
    y=y_val,
    classifier=classifier,
    attack=attack,
    patch=patch,
    n=10
)

