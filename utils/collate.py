def custom_collate(batch):
    imgs = []
    targets = []
    for item in batch:
        img, target = item
        if img is not None and target is not None:
            imgs.append(img)
            targets.append(target)
    return imgs, targets