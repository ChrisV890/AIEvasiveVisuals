#Deprecated now, original Idea of trying to use ultralytics YOLO model. Instead decided to use ResNet for simplicity and easier training. It would fit in better with my ideas and pipeline rather than this model. Please Refer to CNNM.py, ResNetM.py, and attackUtility.py





# from ultralytics import YOLO
# import cv2
# import numpy as np

# # Load pretrained YOLO model
# model = YOLO("yolov8n.pt")

# # Load image
# image_path = "pImageData/personTest2.jpg"
# original_img = cv2.imread(image_path)

# # Make a copy for patching
# patched_img = original_img.copy()

# # Run detection on original image
# results = model(original_img)
# r = results[0]

# boxes = r.boxes.xyxy
# classes = r.boxes.cls
# confs = r.boxes.conf

# person_boxes = []

# print("=== ORIGINAL DETECTIONS ===")

# for box, cls, conf in zip(boxes, classes, confs):
#     x1, y1, x2, y2 = map(int, box)

#     print(f"Class: {int(cls)}, Confidence: {float(conf):.3f}")
#     print(f"Box: {x1, y1, x2, y2}")

#     if int(cls) == 0:  # person
#         print("Person detected")
#         person_boxes.append((x1, y1, x2, y2))

# print("\nPerson Boxes:", person_boxes)




# # =========================
# # APPLY RESNET PATCH
# # =========================

# # Convert learned patch (torch → numpy)
# patch_np = patch.detach().cpu().permute(1, 2, 0).numpy()
# patch_np = (patch_np * 255).astype(np.uint8)

# for (x1, y1, x2, y2) in person_boxes:

#     person_width = x2 - x1
#     person_height = y2 - y1

#     # Patch size (50% of person)
#     patch_w = int(0.5 * person_width)
#     patch_h = int(0.5 * person_height)

#     # Resize learned patch to match person area
#     resized_patch = cv2.resize(patch_np, (patch_w, patch_h))

#     # Center placement (torso region)
#     px1 = x1 + person_width // 2 - patch_w // 2
#     py1 = y1 + person_height // 2 - patch_h // 2
#     px2 = px1 + patch_w
#     py2 = py1 + patch_h

#     # Safety clamp (VERY IMPORTANT)
#     h, w, _ = patched_img.shape

#     px1 = max(0, px1)
#     py1 = max(0, py1)
#     px2 = min(w, px2)
#     py2 = min(h, py2)

#     # Apply patch
#     patched_img[py1:py2, px1:px2] = resized_patch










# # =========================
# # RUN DETECTION AGAIN
# # =========================

# patched_results = model(patched_img)
# r2 = patched_results[0]

# boxes2 = r2.boxes.xyxy
# classes2 = r2.boxes.cls
# confs2 = r2.boxes.conf

# print("\n=== PATCHED DETECTIONS ===")

# for box, cls, conf in zip(boxes2, classes2, confs2):
#     print(f"Class: {int(cls)}, Confidence: {float(conf):.3f}")










# annotated_original = results[0].plot()
# annotated_patched = patched_results[0].plot()

# cv2.imshow("Original Detection", annotated_original)
# cv2.imshow("Patched Detection", annotated_patched)

# cv2.waitKey(0)
# cv2.destroyAllWindows()