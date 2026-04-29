import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from art.estimators.classification import PyTorchClassifier
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from attackUtility import loader_to_numpy, generate_adversarial_patch, softmax_np, show_clean_vs_patched, show_least_confident_patched_with_clean
from art.attacks.evasion import AdversarialPatch



#CNN Class Defiunition

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(SimpleCNN, self).__init__()

        # --- Convolutional feature extractor ---
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),   # (3,224,224) -> (32,224,224)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),                              # -> (32,112,112)

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # -> (64,112,112)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),                              # -> (64,56,56)

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1), # -> (128,56,56)
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),                              # -> (128,28,28)

            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),# -> (256,28,28)
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),                              # -> (256,14,14)
        )

        # --- Classifier head ---
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 14 * 14, 512),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x











#Create Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN(num_classes=2)
model = model.to(device)

#initialize loss and optimizer
loss_CEL = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

#Wrap to ART Classifier
classifier = PyTorchClassifier(
    model=model,
    loss=loss_CEL,
    optimizer=optimizer,
    input_shape=(3,224,224),
    nb_classes=2,
    clip_values=(-3,3),
    )










#Load Data
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),   # converts to [0 1]
    transforms.Normalize(mean = [0.485, 0.456, 0.406], std  = [0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder(
    root = "AIPeopleDataset",
    transform=transform
)
print("Class Mapping:", train_dataset.class_to_idx)


#Split Data, 80% Train 20% Validation
train_size = int(0.8 * len(train_dataset)) #train split size
val_size = len(train_dataset) - train_size #validation split size

train_set, val_set = random_split(train_dataset, [train_size, val_size]) #split


#prep data to send to model
train_loader = DataLoader(train_set, batch_size=16, shuffle=True) #dataloader push
val_loader = DataLoader(val_set, batch_size=16, shuffle=False)

x_train, y_train = loader_to_numpy(train_loader) #Convert to numpy arrays for ART
x_val, y_val = loader_to_numpy(val_loader)










#Train Model
classifier.fit(
    x_train,
    y_train,
    batch_size=16,
    nb_epochs=5,
    verbose=True
)












#Generate and Train Patch on Data
attack = AdversarialPatch(
    classifier=classifier,
    patch_shape=(3, 50, 50),
    rotation_max=45,
    scale_min=0.7,
    scale_max=1.0,
    learning_rate=5.0,
    max_iter=200,
    batch_size=16,
)

patch, mask = generate_adversarial_patch(
    attack,
    train_loader,
    num_classes=2,
    max_batches=5
)










#Evaluate
# ------------------------------------
# Evaluate the SAME validation batch
# clean first, then patched
# ------------------------------------


# Clean predictions
x_eval = x_val
y_eval = y_val


clean_logits = classifier.predict(x_eval)
clean_probs = softmax_np(clean_logits)
clean_pred_classes = np.argmax(clean_probs, axis=1)
clean_confidences = np.max(clean_probs, axis=1)
clean_accuracy = np.mean(clean_pred_classes == y_eval)
clean_true_conf = clean_probs[np.arange(len(y_eval)), y_eval]






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
patched_true_conf = patched_probs[np.arange(len(y_eval)), y_eval]




# Metrics
print("\n--- Same Batch Comparison ---")
print("Clean Accuracy:", clean_accuracy)
print("Patched Accuracy:", patched_accuracy)
print("Accuracy Drop:", clean_accuracy - patched_accuracy)
print("Mean Clean Confidence:", np.mean(clean_confidences))
print("Mean Patched Confidence:", np.mean(patched_confidences))
print("Mean Confidence Drop:", np.mean(clean_confidences) - np.mean(patched_confidences))
print("Mean True-Class Confidence (clean):", np.mean(clean_true_conf))
print("Mean True-Class Confidence (patched):", np.mean(patched_true_conf))
print("True-Class Confidence Drop:", np.mean(clean_true_conf) - np.mean(patched_true_conf))
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

