import numpy as np
import torch
import matplotlib.pyplot as plt
from art.attacks.evasion import AdversarialPatch

#Generate Adverial pAtch
def to_one_hot(y, num_classes):
    return np.eye(num_classes)[y].astype(np.float32)

def get_mixed_batch(data_loader):
    """
    Returns the first batch containing both classes.
    This makes patch generation more meaningful than using a single-class batch.
    """
    for images, labels in data_loader:
        y = labels.numpy().astype(np.int64)
        if len(np.unique(y)) > 1:
            x = images.numpy().astype(np.float32)
            return x, y

    # Fallback: just return the first batch if no mixed batch is found
    images, labels = next(iter(data_loader))
    return images.numpy().astype(np.float32), labels.numpy().astype(np.int64)

def generate_adversarial_patch(attack, data_loader, num_classes=2):
    """
    Trains the ART adversarial patch and returns:
    - patch: the learned patch
    - mask: the returned mask from ART
    - x_batch: the batch used to generate the patch
    - y_batch: the labels used to generate the patch
    """
    x_batch, y_batch = get_mixed_batch(data_loader)
    y_batch_onehot = to_one_hot(y_batch, num_classes=num_classes)

    patch, mask = attack.generate(x=x_batch, y=y_batch_onehot)

    print("Patch generated.")
    print("Patch shape:", patch.shape)
    print("Mask shape:", mask.shape)

    return patch, mask, x_batch, y_batch



#Patch Application
def apply_patch_to_batch(attack, x_batch, patch, scale=0.4):
    """
    Applies a learned adversarial patch to a batch of images.
    This keeps things simple by not passing a mask.
    """
    x_patched = attack.apply_patch(
        x_batch,
        scale=scale,
        patch_external=patch
    )
    return x_patched


#Evaluation on clean or patch data
def softmax_np(logits):
    exps = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    return exps / np.sum(exps, axis=1, keepdims=True)

def evaluate_loader(classifier, data_loader, attack=None, patch=None, scale=0.4, patched=False):
    """
    Evaluates the classifier on a loader.
    If patched=True and patch is provided, the patch is applied before prediction.
    
    Prints:
    - accuracy
    - predicted class counts
    - true class counts
    - sample confidences
    - average loss
    """
    all_preds = []
    all_labels = []
    all_confidences = []
    all_losses = []

    for images, labels in data_loader:
        x = images.numpy().astype(np.float32)
        y = labels.numpy().astype(np.int64)

        if patched:
            if attack is None or patch is None:
                raise ValueError("patched=True requires both attack and patch.")
            x = attack.apply_patch(
                x,
                scale=scale,
                patch_external=patch
            )

        preds = classifier.predict(x)              # ART returns model outputs
        pred_classes = np.argmax(preds, axis=1)

        probs = softmax_np(preds)
        confidences = np.max(probs, axis=1)

        losses = classifier.compute_loss(x, y)     # labels can be indices or one-hot

        all_preds.append(pred_classes)
        all_labels.append(y)
        all_confidences.append(confidences)
        all_losses.append(losses)

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    all_confidences = np.concatenate(all_confidences)
    all_losses = np.concatenate(all_losses)

    accuracy = np.mean(all_preds == all_labels)

    print("Validation Accuracy (patched)" if patched else "Validation Accuracy (clean):", accuracy)
    print("Predicted class counts:", np.bincount(all_preds))
    print("True class counts:", np.bincount(all_labels))
    print("Sample confidences:", all_confidences[:10])
    print("Average loss:", np.mean(all_losses))

    return {
        "accuracy": accuracy,
        "predicted_classes": all_preds,
        "true_classes": all_labels,
        "confidences": all_confidences,
        "losses": all_losses,
        "average_loss": np.mean(all_losses),
    }



#only unnormalizes to show patch
def unnormalize_image(x):
    """
    x: CHW normalized tensor/array
    returns: HWC image in [0, 1] for display
    """
    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


    if isinstance(x, np.ndarray):
        img = x.copy()
    else:
        img = x.detach().cpu().numpy().copy()

    # CHW -> HWC
    img = np.transpose(img, (1, 2, 0))

    # reverse ImageNet normalization
    img = (img * IMAGENET_STD) + IMAGENET_MEAN

    # clip for display
    img = np.clip(img, 0, 1)
    return img




def show_original_and_patched(classifier, attack, data_loader, patch, scale=0.4, index=0):
    """
    Displays one original image and the same image with the patch applied.
    """
    images, labels = next(iter(data_loader))
    x = images.numpy().astype(np.float32)

    if index >= len(x):
        raise IndexError("index is out of range for the current batch.")

    original = x[index:index+1]  # keep batch dimension
    patched = attack.apply_patch(
        original,
        scale=scale,
        patch_external=patch
    )

    orig_img = unnormalize_image(original[0])
    patched_img = unnormalize_image(patched[0])

    orig_pred = np.argmax(classifier.predict(original), axis=1)[0]
    patched_pred = np.argmax(classifier.predict(patched), axis=1)[0]

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(orig_img)
    plt.title(f"Original | True: {labels[index].item()} | Pred: {orig_pred}")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(patched_img)
    plt.title(f"Patched | Pred: {patched_pred}")
    plt.axis("off")

    plt.tight_layout()
    plt.show()



def rect_coords(x, y, w, h):
    return np.array([
        [x,     y],
        [x + w, y],
        [x + w, y + h],
        [x,     y + h],
    ], dtype=np.float32)



def torso_patch_coords(bbox, patch_w, patch_h, img_w, img_h, torso_y_frac=0.45):
    x1, y1, x2, y2 = bbox
    person_w = x2 - x1
    person_h = y2 - y1

    cx = int((x1 + x2) / 2)
    cy = int(y1 + torso_y_frac * person_h)

    ulx = max(0, min(img_w - patch_w, cx - patch_w // 2))
    uly = max(0, min(img_h - patch_h, cy - patch_h // 2))

    return rect_coords(ulx, uly, patch_w, patch_h)




def apply_patch_on_person(image_chw, patch_chw, bbox, attack):
    """
    image_chw: (C, H, W)
    patch_chw: (C, ph, pw)
    bbox: (x1, y1, x2, y2)
    """
    _, h, w = image_chw.shape
    _, ph, pw = patch_chw.shape

    coords = torso_patch_coords(
        bbox=bbox,
        patch_w=pw,
        patch_h=ph,
        img_w=w,
        img_h=h,
        torso_y_frac=0.45
    )

    return attack.insert_transformed_patch(
        image_chw,
        patch_chw,
        coords
    )




def apply_patch_to_batch_with_bbox(x_batch, patch, bbox_fn, attack):
    patched_batch = []

    for i in range(x_batch.shape[0]):
        img = x_batch[i]
        bbox = bbox_fn(img)

        patched_img = apply_patch_on_person(
            image_chw=img,
            patch_chw=patch,
            bbox=bbox,
            attack=attack
        )

        patched_batch.append(patched_img)

    return np.stack(patched_batch)


def get_center_torso_bbox(img):
    """
    img: (C, H, W)
    Returns a bbox roughly where a torso would be
    """
    _, h, w = img.shape

    x1 = int(w * 0.3)
    x2 = int(w * 0.7)
    y1 = int(h * 0.3)
    y2 = int(h * 0.75)

    return (x1, y1, x2, y2)



def evaluate_with_realistic_patch(classifier, val_loader, patch, attack):
    all_preds = []
    all_labels = []

    for images, labels in val_loader:
        x = images.numpy().astype(np.float32)
        y = labels.numpy().astype(np.int64)

        x_patched = apply_patch_to_batch_with_bbox(
            x_batch=x,
            patch=patch,
            bbox_fn=get_center_torso_bbox,
            attack=attack
        )

        preds = classifier.predict(x_patched)
        pred_classes = np.argmax(preds, axis=1)

        all_preds.append(pred_classes)
        all_labels.append(y)

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    accuracy = np.mean(all_preds == all_labels)

    print("Validation Accuracy (realistic patched):", accuracy)
    print("Predicted class counts:", np.bincount(all_preds))
    print("True class counts:", np.bincount(all_labels))

    return accuracy