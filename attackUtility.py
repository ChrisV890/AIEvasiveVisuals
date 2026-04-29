import numpy as np
import math
import matplotlib.pyplot as plt
#from art.attacks.evasion import AdversarialPatch

def loader_to_numpy(loader):
    x_list = []
    y_list = []

    for images, labels in loader:
        x_list.append(images.numpy())   # already [0, 1] and NCHW from ToTensor()
        y_list.append(labels.numpy())

    x = np.concatenate(x_list, axis=0).astype(np.float32)
    y = np.concatenate(y_list, axis=0).astype(np.int64)
    return x, y


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

def generate_adversarial_patch(attack, data_loader, num_classes=2, max_batches=10):
    """
    Train adversarial patch using MULTIPLE batches instead of just one.

    max_batches: how many batches to use (increase for stronger patch)
    """

    x_list = []
    y_list = []

    for i, (images, labels) in enumerate(data_loader):
        if i >= max_batches:
            break

        x = images.numpy().astype(np.float32)
        y = labels.numpy().astype(np.int64)

        x_list.append(x)
        y_list.append(y)

    # Combine batches
    x_all = np.concatenate(x_list, axis=0)
    y_all = np.concatenate(y_list, axis=0)

    #y_all_onehot = to_one_hot(y_all, num_classes=num_classes)

    patch, mask = attack.generate(x=x_all, y=y_all)

    print("Patch generated.")
    print("Patch shape:", patch.shape)
    print("Mask shape:", mask.shape)
    print("Total images used for patch:", len(x_all))

    return patch, mask


#Evaluation on clean or patch data
def softmax_np(logits):
    logits = np.asarray(logits)
    exp = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    return exp / np.sum(exp, axis=1, keepdims=True)


#only unnormalizes to show patch
def unnormalize_image(img_chw):

    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    """
    img_chw: (C, H, W), normalized tensor/array
    returns: (H, W, C) in [0, 1] for display
    """
    img = np.array(img_chw, copy=True)
    img = np.transpose(img, (1, 2, 0))
    img = img * IMAGENET_STD + IMAGENET_MEAN
    img = np.clip(img, 0, 1)
    return img


# def show_clean_vs_patched(x, y, attack, patch, classifier=None, n=10, scale=0.4):
#     n = min(n, len(x))
#     x_clean = x[:n]
#     y_clean = y[:n]

#     x_patched = attack.apply_patch(
#         x_clean,
#         scale=scale,
#         patch_external=patch
#     )

#     plt.figure(figsize=(16, 8 * n))

#     for i in range(n):
#         clean_img = unnormalize_image(x_clean[i])
#         patched_img = unnormalize_image(x_patched[i])

#         if classifier is not None:
#             clean_pred = np.argmax(classifier.predict(x_clean[i:i+1]), axis=1)[0]
#             patched_pred = np.argmax(classifier.predict(x_patched[i:i+1]), axis=1)[0]
#             clean_title = f"Clean\nTrue: {y_clean[i]} Pred: {clean_pred}"
#             patched_title = f"Patched\nPred: {patched_pred}"
#         else:
#             clean_title = f"Clean\nTrue: {y_clean[i]}"
#             patched_title = "Patched"

#         plt.subplot(n, 2, 2 * i + 1)
#         plt.imshow(clean_img)
#         plt.title(clean_title)
#         plt.axis("off")

#         plt.subplot(n, 2, 2 * i + 2)
#         plt.imshow(patched_img)
#         plt.title(patched_title)
#         plt.axis("off")

#     plt.tight_layout()
#     plt.show()



# def show_least_confident_patched_with_clean(
#     x, y, classifier, attack, patch, n=10, scale=0.4
# ):
#     """
#     Shows the N images where the PATCHED prediction has the lowest confidence,
#     alongside their CLEAN versions.

#     Displays:
#     - Clean image + confidence
#     - Patched image + confidence
#     """

#     # --- Clean predictions ---
#     clean_logits = classifier.predict(x)
#     clean_probs = softmax_np(clean_logits)
#     clean_preds = np.argmax(clean_probs, axis=1)
#     clean_conf = np.max(clean_probs, axis=1)

#     # --- Patched predictions ---
#     x_patched = attack.apply_patch(
#         x,
#         scale=scale,
#         patch_external=patch
#     )

#     patched_logits = classifier.predict(x_patched)
#     patched_probs = softmax_np(patched_logits)
#     patched_preds = np.argmax(patched_probs, axis=1)
#     patched_conf = np.max(patched_probs, axis=1)

#     # --- Find lowest-confidence patched indices ---
#     idxs = np.argsort(patched_conf)[:n]

#     # --- Plot ---
#     rows = n
#     cols = 2
#     plt.figure(figsize=(16, 8 * rows))

#     for i, idx in enumerate(idxs):
#         clean_img = unnormalize_image(x[idx])
#         patched_img = unnormalize_image(x_patched[idx])

#         # Clean
#         plt.subplot(rows, cols, 2*i + 1)
#         plt.imshow(clean_img)
#         plt.title(
#             f"CLEAN\nT:{y[idx]} P:{clean_preds[idx]}\nConf:{clean_conf[idx]:.3f}",
#             fontsize=9
#         )
#         plt.axis("off")

#         # Patched
#         plt.subplot(rows, cols, 2*i + 2)
#         plt.imshow(patched_img)
#         plt.title(
#             f"PATCHED\nP:{patched_preds[idx]}\nConf:{patched_conf[idx]:.3f}",
#             fontsize=9
#         )
#         plt.axis("off")

#     plt.suptitle("Lowest Confidence Patched Images (with Clean Comparison)", fontsize=14)
#     plt.tight_layout()
#     plt.show()




def show_clean_vs_patched(x, y, attack, patch, classifier=None, n=10, scale=0.4):
    n = min(n, len(x))
    x_clean = x[:n]
    y_clean = y[:n]

    x_patched = attack.apply_patch(
        x_clean,
        scale=scale,
        patch_external=patch
    )

    cols = int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(n / cols))

    # CLEAN WINDOW
    fig1, axes1 = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows), dpi=150)
    axes1 = np.array(axes1).reshape(-1)

    for i in range(n):
        clean_img = unnormalize_image(x_clean[i])

        if classifier is not None:
            clean_pred = np.argmax(classifier.predict(x_clean[i:i+1]), axis=1)[0]
            title = f"True: {y_clean[i]} | Pred: {clean_pred}"
        else:
            title = f"True: {y_clean[i]}"

        axes1[i].imshow(clean_img)
        axes1[i].set_title(title, fontsize=10)
        axes1[i].axis("off")

    for j in range(n, len(axes1)):
        axes1[j].axis("off")

    fig1.suptitle("CLEAN IMAGES", fontsize=18)
    plt.tight_layout()
    #plt.show()

    # PATCHED WINDOW
    fig2, axes2 = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows), dpi=150)
    axes2 = np.array(axes2).reshape(-1)

    for i in range(n):
        patched_img = unnormalize_image(x_patched[i])

        if classifier is not None:
            patched_pred = np.argmax(classifier.predict(x_patched[i:i+1]), axis=1)[0]
            title = f"Pred: {patched_pred}"
        else:
            title = "Patched"

        axes2[i].imshow(patched_img)
        axes2[i].set_title(title, fontsize=10)
        axes2[i].axis("off")

    for j in range(n, len(axes2)):
        axes2[j].axis("off")

    fig2.suptitle("PATCHED IMAGES", fontsize=18)
    plt.tight_layout()
    plt.show()














def show_least_confident_patched_with_clean(x, y, classifier, attack, patch, n=10, scale=0.4):
    clean_logits = classifier.predict(x)
    clean_probs = softmax_np(clean_logits)
    clean_preds = np.argmax(clean_probs, axis=1)
    clean_conf = np.max(clean_probs, axis=1)

    x_patched = attack.apply_patch(
        x,
        scale=scale,
        patch_external=patch
    )

    patched_logits = classifier.predict(x_patched)
    patched_probs = softmax_np(patched_logits)
    patched_preds = np.argmax(patched_probs, axis=1)
    patched_conf = np.max(patched_probs, axis=1)

    idxs = np.argsort(patched_conf)[:n]

    cols = int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(n / cols))

    # CLEAN WINDOW
    fig1, axes1 = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows), dpi=150)
    axes1 = np.array(axes1).reshape(-1)

    for i, idx in enumerate(idxs):
        clean_img = unnormalize_image(x[idx])

        axes1[i].imshow(clean_img)
        axes1[i].set_title(
            f"T:{y[idx]} | P:{clean_preds[idx]}\nConf:{clean_conf[idx]:.3f}",
            fontsize=10
        )
        axes1[i].axis("off")

    for j in range(n, len(axes1)):
        axes1[j].axis("off")

    fig1.suptitle("CLEAN IMAGES (Lowest Confidence Patched Set)", fontsize=18)
    plt.tight_layout()
    #plt.show()

    # PATCHED WINDOW
    fig2, axes2 = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows), dpi=150)
    axes2 = np.array(axes2).reshape(-1)

    for i, idx in enumerate(idxs):
        patched_img = unnormalize_image(x_patched[idx])

        axes2[i].imshow(patched_img)
        axes2[i].set_title(
            f"P:{patched_preds[idx]}\nConf:{patched_conf[idx]:.3f}",
            fontsize=10
        )
        axes2[i].axis("off")

    for j in range(n, len(axes2)):
        axes2[j].axis("off")

    fig2.suptitle("PATCHED IMAGES (Lowest Confidence)", fontsize=18)
    plt.tight_layout()
    plt.show()