import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import AdversarialPatch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

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





#Maybe not needed
def preprocess(images):
    x = np.array(images)
    x = x.transpose(0, 3, 1, 2)  # NHWC → NCHW
    x = x / 255.0
    return x.astype(np.float32)


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
    clip_values=(0, 1),
)
print("Classifier Primed\n")




#------------------------------------
#Adverserial Patch Attack
#------------------------------------

attack = AdversarialPatch(
    classifier=classifier,
    patch_shape=(3, 50, 50),  #i dont fucking know if this is good.
    rotation_max=22.5,
    scale_min=0.5,
    scale_max=1.0,
    learning_rate=5.0,
    max_iter=100,
    batch_size=8,
)
print("Adverserial Patch Attack Primed\n")




#------------------------------------
#Dataset Billdin
#------------------------------------


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),   # converts to [0 1]
])

train_dataset = datasets.ImageFolder(
    root="testtpeeple/Human Faces Dataset",   #EXTREMELY TEMPORARY I SWEAR
    transform=transform
)
print("Dataset Primed\n")

train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True
)
print("Data Loaded\n")


#--------------------------------------------------------------
#DELETEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE
#data to numpy for ART
#x_batch, y_batch = next(iter(train_loader))

#x = x_batch.numpy().astype(np.float32)   # NO transpose
#y = y_batch.numpy()

#y_onehot = np.eye(2)[y]

#print(x.shape)

#patch, _ = attack.generate(x=x, y=y_onehot)
#--------------------------------------------------------------



#------------------------------------
#1. Train Classifier on cleaned data
#------------------------------------
x_train, y_train = loader_to_numpy(train_loader)

#small data so i can get through this shit quicker
x_train_small = x_train[:128]
y_train_small = y_train[:128]


classifier.fit(
    x_train_small,
    y_train_small,
    batch_size=16,
    nb_epoch=1,
    verbose=True
)

#----------------------------------------------
#Misleading as hell, Ive only got one class, confidence only produce logits, still need to softmax that
# Get one batch from your loader
#x_clean, y_clean = next(iter(train_loader))
#x_clean = x_clean.numpy().astype(np.float32)
#y_clean = y_clean.numpy().astype(np.int64)

# Predict
#preds = classifier.predict(x_clean)

# Predicted class for each image
#pred_classes = np.argmax(preds, axis=1)

# Confidence for predicted class
#pred_confidence = np.max(preds, axis=1)

#print("Predicted classes:", pred_classes[:20])
#print("True classes:", y_clean[:20])
#print("Confidences:", pred_confidence[:20])

# Accuracy on this batch
#accuracy = np.mean(pred_classes == y_clean)
#print("Batch accuracy:", accuracy)
#----------------------------------------------



#------------------------------------
#2. Generate Adverserial Patch from one batch
#------------------------------------
x_batch, y_batch = next(iter(train_loader))
x_batch = x_batch.numpy().astype(np.float32)
y_batch = y_batch.numpy().astype(np.int64)

y_batch_onehot = to_one_hot(y_batch, num_classes=2)

patch,_ = attack.generate(x=x_batch, y=y_batch_onehot)


print("Patch shape:", patch.shape)



#------------------------------------
#3. Apply the patch to images
#------------------------------------
x_patched = attack.apply_patch(
    x_batch,
    scale=0.4,
    patch_external=patch
)
print("Patches Applied WE DID IT")


clean_preds = classifier.predict(x_batch)
patched_preds = classifier.predict(x_patched)

clean_loss = classifier.compute_loss(x_batch, y_batch_onehot)
patched_loss = classifier.compute_loss(x_patched, y_batch_onehot)


print("Clean predictions (first 5):")
print(clean_preds[:5])

print("Patched predictions (first 5):")
print(patched_preds[:5])

print("Clean loss (first 5):")
print(clean_loss[:5])

print("Patched loss (first 5):")
print(patched_loss[:5])

print("Clean predicted classes:", np.argmax(clean_preds, axis=1)[:20])
print("Patched predicted classes:", np.argmax(patched_preds, axis=1)[:20])













