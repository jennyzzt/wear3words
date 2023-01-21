import cv2
import os
import json
import numpy as np
import rembg


def init_measurements_range():
    MSMTS_CONTEXT = {
        'unit': 'inches',
        'step': 0.1,
    }
    MSMTS_RANGE = {
        'chest': (33.0, 47.0),  # (30.0, 64.0)
        'waist': (25.0, 40.0),  # (22.0, 61.0)
        'hip': (35.0, 49.0),    # (32.0, 64.0)
        # 'inseam': (28.0, 36.0)
    }

    with open('./words.json', 'r') as f:
        WORDS = json.load(f)['words']
        np.random.seed(5)
        np.random.shuffle(WORDS)

    return MSMTS_CONTEXT, MSMTS_RANGE, WORDS


MSMTS_CONTEXT, MSMTS_RANGE, WORDS = init_measurements_range()


def measurements_to_wordidx(body_msmts):
    # one-one matching between body measure index to word combination index
    body_idx = {k: (v - MSMTS_RANGE[k][0]) / MSMTS_CONTEXT['step'] for k, v in body_msmts.items()}
    msmts_steps = {k: (v[1] - v[0]) / MSMTS_CONTEXT['step'] + 1 for k, v in MSMTS_RANGE.items()}
    word_combi_idx = 0
    for i, (k, v) in enumerate(body_idx.items()):
        if i == len(msmts_steps) - 1:
            word_combi_idx += v
        else:
            word_combi_idx += v * sum(list(msmts_steps.values())[i+1:])
    word_combi_idx = round(word_combi_idx, 0)

    # unpack word combination index to invidual word indices
    word_idx = np.zeros(3)
    subcombi_length = len(WORDS) - len(word_idx) + 1
    for i in range(len(word_idx)):
        step_size = subcombi_length ** (len(word_idx) - i - 1)
        word_idx[i] = word_combi_idx // step_size
        word_combi_idx -= word_idx[i] * step_size

    # shift the words idx if previous words are taken
    for i, idx in enumerate(word_idx):
        smaller_occs = (word_idx[:i] <= idx).sum()
        word_idx[i] += smaller_occs

    return word_idx


def measurements_to_words(body_msmts):
    word_idx = measurements_to_wordidx(body_msmts)
    words = [WORDS[int(idx)] for idx in word_idx]
    return words


def image_to_measurements(filepath, filepath_side, bodylen):
    # use openpose to estimate keypoints
    os.system('cd ../openpose && ./build/examples/openpose/openpose.bin --image_dir ../wear3words/uploaded_images/ --write_json ../wear3words/uploaded_images/processed/ --write_images ../wear3words/uploaded_images/processed/')

    # remove image background and save
    img = cv2.imread(filepath)
    img_output = rembg.remove(img)
    cv2.imwrite('./uploaded_images/processed/front_nobg.png', img_output)
    imgside = cv2.imread(filepath_side)
    imgside_output = rembg.remove(imgside)
    cv2.imwrite('./uploaded_images/processed/side_nobg.png', imgside_output)

    # estimate measurements from openpose keypoints
    with open('./uploaded_images/processed/person_front_keypoints.json', 'r') as f:
        data = json.load(f)
    keypoints = data['people'][0]['pose_keypoints_2d']
    with open('./uploaded_images/processed/person_side_keypoints.json', 'r') as f:
        data = json.load(f)
    keypoints_side = data['people'][0]['pose_keypoints_2d']

    # util functions
    def get_keypointxy(x, sideview=False):
        if sideview:
            return np.array(keypoints_side[x*3:x*3+2])
        return np.array(keypoints[x*3:x*3+2])

    def get_dist(x, y):
        return np.linalg.norm(x - y)

    def get_oval_circum(d1, d2):
        return np.pi * np.sqrt((np.power(d1, 2) + np.power(d2, 2))/ 2)

    def save_viz_points(img, coords, name):
        newimg = img.copy()
        for coord in coords:
            newimg = cv2.circle(newimg, (int(coord[0]), int(coord[1])), radius=20, color=(0, 0, 255, 255), thickness=-1)
        cv2.imwrite(f'./uploaded_images/processed/{name}.png', newimg)

    # image to real estimate scale by body length
    height_est = bodylen
    # front estimate
    height_img = get_dist(get_keypointxy(11), get_keypointxy(9)) + \
      get_dist(get_keypointxy(8), get_keypointxy(1))
    img_est_scale = height_est / height_img
    # side estimate
    height_img_side = get_dist(get_keypointxy(11, sideview=True), get_keypointxy(9, sideview=True)) + \
      get_dist(get_keypointxy(8, sideview=True), get_keypointxy(1, sideview=True))
    imgside_est_scale = height_est / height_img_side

    # front chest estimate
    chest_left = get_keypointxy(2)
    chest_right = get_keypointxy(5)
    chest_mid = get_keypointxy(1)
    chest_leftvec = (chest_left - chest_mid) / get_dist(chest_left, chest_mid)
    chest_rightvec = (chest_right - chest_mid) / get_dist(chest_right, chest_mid)
    chest_frontlength = get_dist(chest_right, chest_left)
    # visualize front chest estimate
    save_viz_points(img_output, [chest_left, chest_right], 'chest_front_est')
    # side chest estimate
    chest_side_mid = get_keypointxy(1, sideview=True)
    chest_side_mid[1] += 4.0/imgside_est_scale
    chest_side_leftvec = [-1.0, 0.0]
    chest_side_rightvec = [1.0, 0]
    chest_side_leftend = chest_side_mid.copy()
    chest_side_rightend = chest_side_mid.copy()
    while imgside_output[int(chest_side_leftend[1]), int(chest_side_leftend[0])][-1] != 0:
      chest_side_leftend += chest_side_leftvec
    while imgside_output[int(chest_side_rightend[1]), int(chest_side_rightend[0])][-1] != 0:
      chest_side_rightend += chest_side_rightvec
    chest_sidelength = get_dist(chest_side_rightend, chest_side_leftend)
    # visualize side chest estimate
    save_viz_points(imgside_output, [chest_side_leftend, chest_side_rightend], 'chest_side_est')
    # chest circumference estimate
    chest_est = get_oval_circum(chest_frontlength * img_est_scale, chest_sidelength * imgside_est_scale)

    # front hip estimate
    hip_left = get_keypointxy(9)
    hip_right = get_keypointxy(12)
    hip_mid = get_keypointxy(8)
    hip_leftvec = (hip_left - hip_mid) / get_dist(hip_left, hip_mid)
    hip_leftend = hip_left.copy()
    while img_output[int(hip_leftend[1]), int(hip_leftend[0])][-1] != 0:
      hip_leftend += hip_leftvec
    hip_rightvec = (hip_right - hip_mid) / get_dist(hip_right, hip_mid)
    hip_rightend = hip_right.copy()
    while img_output[int(hip_rightend[1]), int(hip_rightend[0])][-1] != 0:
      hip_rightend += hip_rightvec
    hip_frontlength = get_dist(hip_rightend, hip_leftend)
    # visualize front hip estimate
    save_viz_points(img_output, [hip_leftend, hip_rightend], 'hip_front_est')
    # side hip estimate
    hip_side_mid = get_keypointxy(8, sideview=True)
    hip_side_leftvec = [-1.0, 0.0]
    hip_side_rightvec = [1.0, 0]
    hip_side_leftend = hip_side_mid.copy()
    hip_side_rightend = hip_side_mid.copy()
    while imgside_output[int(hip_side_leftend[1]), int(hip_side_leftend[0])][-1] != 0:
      hip_side_leftend += hip_side_leftvec
    while imgside_output[int(hip_side_rightend[1]), int(hip_side_rightend[0])][-1] != 0:
      hip_side_rightend += hip_side_rightvec
    hip_sidelength = get_dist(hip_side_rightend, hip_side_leftend)
    # visualize side hip estimate
    save_viz_points(imgside_output, [hip_side_leftend, hip_side_rightend], 'hip_side_est')
    # hip circumference estimate
    hip_est = get_oval_circum(hip_frontlength * img_est_scale, hip_sidelength * imgside_est_scale)

    # front waist estimate
    waist_mid = chest_mid + (hip_mid - chest_mid) / 2
    waist_leftvec = (chest_leftvec + hip_leftvec) / 2
    waist_rightvec = (chest_rightvec + hip_rightvec) / 2
    waist_leftend = waist_mid.copy()
    while img_output[int(waist_leftend[1]), int(waist_leftend[0])][-1] != 0:
      waist_leftend += waist_leftvec
    waist_rightend = waist_mid.copy()
    while img_output[int(waist_rightend[1]), int(waist_rightend[0])][-1] != 0:
      waist_rightend += waist_rightvec
    waist_frontlength = get_dist(waist_rightend, waist_leftend)
    # visualize front waist estimate
    save_viz_points(img_output, [waist_leftend, waist_rightend], 'waist_front_est')
    # side waist estimate
    waist_side_mid = chest_side_mid + (hip_side_mid - chest_side_mid) / 2
    waist_side_leftvec = [-1.0, 0.0]
    waist_side_rightvec = [1.0, 0]
    waist_side_leftend = waist_side_mid.copy()
    waist_side_rightend = waist_side_mid.copy()
    while imgside_output[int(waist_side_leftend[1]), int(waist_side_leftend[0])][-1] != 0:
      waist_side_leftend += waist_side_leftvec
    while imgside_output[int(waist_side_rightend[1]), int(waist_side_rightend[0])][-1] != 0:
      waist_side_rightend += waist_side_rightvec
    waist_sidelength = get_dist(waist_side_rightend, waist_side_leftend)
    # visualize front waist estimate
    save_viz_points(imgside_output, [waist_side_leftend, waist_side_rightend], 'waist_side_est')
    # waist circumference estimate
    waist_est = get_oval_circum(waist_frontlength * img_est_scale, waist_sidelength * imgside_est_scale)

    # return measurements
    msmts_est = {
        'chest': round(chest_est, 1),
        'waist': round(waist_est, 1),
        'hip': round(hip_est, 1),
    }
    return msmts_est


if __name__ == "__main__":
    print('Hello!')
    body_msmts = {  # dummy values
        'chest': 35.0,
        'waist': 27.4,
        'hip': 37.1,
    }
    # get body measurements input
    for k in MSMTS_RANGE.keys():
        success = False
        while not success:
            val = float(input(f'Enter your {k} size in {MSMTS_CONTEXT["unit"]}: '))
            if val < MSMTS_RANGE[k][0] or val > MSMTS_RANGE[k][1]:
                print(f'Sorry, we only support a range of {MSMTS_RANGE[k][0]} to {MSMTS_RANGE[k][1]} '
                      f'inches for {k} measurement. Please try again.\n')
            else:
                success = True
        body_msmts[k] = val
        success = True
    print(f'Your measurements: {body_msmts}')

    words = measurements_to_words(body_msmts)
    print(f'///{".".join(words)}')
