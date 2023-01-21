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
    # TODO: dummy values
    msmts_est = {
        'chest': 35.0,
        'waist': 35.0,
        'hip': 35.0,
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
