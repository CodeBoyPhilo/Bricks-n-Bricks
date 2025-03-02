import concurrent.futures

import cv2
import numpy as np


def load_image(img_path):

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    return img, img_gray


def segment_objects(img, img_gray):
    _, thresh = cv2.threshold(img_gray, 200, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bounds = [cv2.boundingRect(c) for c in contours]
    sorted_contours = sorted(zip(contours, bounds), key=lambda x: (x[1][1], x[1][0]))

    icons = []
    for contour, (x, y, w, h) in sorted_contours:
        if cv2.contourArea(contour) > 500:
            icon = img[y : y + h, x : x + w]
            icons.append(icon)

    return icons


def resize_icon(icon, target_size=(118, 116)):
    resized_icon = cv2.resize(icon, target_size)
    return resized_icon


def template_matching(icons, threshold=0.95):
    clusters = []
    visited = [False] * len(icons)
    labels = [-1] * len(icons)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        resized_icons = list(executor.map(resize_icon, icons))

    def _assign_cluster(icon, icon_index):
        for cluster_idx, cluster in enumerate(clusters):
            result = cv2.matchTemplate(
                resized_icons[cluster[0]], icon, cv2.TM_CCOEFF_NORMED
            )
            _, max_val, _, _ = cv2.minMaxLoc(result)
            if max_val >= threshold:
                cluster.append(icon_index)
                labels[icon_index] = cluster_idx
                return True
        return False

    def process_icon(i):
        if visited[i]:
            return
        if not _assign_cluster(resized_icons[i], i):
            clusters.append([i])
            labels[i] = len(clusters) - 1
        visited[i] = True

    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.map(process_icon, range(len(icons)))

    labeled_cluster = {
        cluster_label + 1: cluster for cluster_label, cluster in enumerate(clusters)
    }

    labels = np.array(labels).reshape(14, 10) + 1

    return labeled_cluster, labels
