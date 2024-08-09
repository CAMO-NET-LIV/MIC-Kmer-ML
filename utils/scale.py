def scale_labels(labels, min, max):
    labels = (labels - min) / (max - min) * 2 - 1
    return labels


def descale_labels(labels, min, max):
    try:
        labels = (labels + 1) / 2 * (max - min) + min
    except Exception as e:
        labels = [((label + 1) / 2 * (max - min) + min) for label in labels]
    return labels