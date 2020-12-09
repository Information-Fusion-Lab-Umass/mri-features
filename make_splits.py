
#!/usr/bin/env python3

import argparse
from datetime import datetime
import json
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
DEFAULT_OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'splits')
META_FILE = os.path.join(SCRIPT_DIR, 'mriimg_meta_v4.csv')

np.random.seed(42)

def test_split():
    data_df = pd.read_csv(META_FILE)
    ptids = list(data_df['PTID'].unique())

    freq_dict = {}
    ptid_scores = {}
    actual_labels = {}

    for ptid in ptids:
        query = data_df.loc[data_df["PTID"] == ptid]

        count = [0, 0, 0]  # CN, MCI, AD

        labels = []

        for idx, row in query.iterrows():
            if row["label2"] == "CN":
                count[0] += 1
                labels.append(0)
            elif row["label2"] == "AD":
                count[2] += 1
                labels.append(1)
            else:
                count[1] += 1
                labels.append(2)

        ptid_scores[ptid] = 1 * count[0] + 10 * count[1] + 100 * count[2]
        freq_dict[ptid] = count
        actual_labels[ptid] = labels

    kfold = StratifiedKFold(10, shuffle=True)
    splits = [split for _, split in kfold.split(range(len(ptid_scores)), np.array(list(ptid_scores.values())))]

    # Print metrics for each split to make sure they're stratified
    print('split metrics')
    for split_num, split in enumerate(splits):
        cn = 0
        ad = 0
        mci = 0

        for idx in split:
            labels = actual_labels[list(ptid_scores.keys())[idx]]
            for l in labels:
                if l == 0:
                    cn += 1
                elif l == 1:
                    mci += 1
                elif l == 2:
                    ad += 1

        print('split {}: {} cn {} ad {} mci, {} patients, {} visits'.format(
            split_num, cn, ad, mci, len(split), cn + ad + mci))


def get_patient_label(df_path):
    data_df = pd.read_csv(df_path)
    ptids = list(data_df['PTID'].unique())

    freq_dict = {}
    ptid_scores = {}
    actual_labels = {}

    for ptid in ptids:
        query = data_df.loc[data_df["PTID"] == ptid]

        count = [0, 0, 0]  # CN, MCI, AD

        labels = []

        for idx, row in query.iterrows():
            if row["label2"] == "CN":
                count[0] += 1
                labels.append(0)
            elif row["label2"] == "AD":
                count[2] += 1
                labels.append(1)
            else:
                count[1] += 1
                labels.append(2)

        ptid_scores[ptid] = 1 * count[0] + 10 * count[1] + 100 * count[2]
        freq_dict[ptid] = count
        actual_labels[ptid] = labels

    kfold = StratifiedKFold(10, shuffle=True)
    splits = [split for _, split in kfold.split(range(len(ptid_scores)), np.array(list(ptid_scores.values())))]

    visit_indices = []

    for split_num, split in enumerate(splits):
        split_visits = []
        for idx in split:
            ptid = list(ptid_scores.keys())[idx]
            query = data_df.loc[data_df["PTID"] == ptid]
            split_visits.extend(list(query.index))
        split_visits.sort()
        visit_indices.append(split_visits)

    for idx, s in enumerate(visit_indices):
        print(len(s))
        for i in s:
            for j in range(10):
                if j != idx:
                    assert i not in visit_indices[j]

    return visit_indices


def store_indices():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-splits', type=int, default=10)
    parser.add_argument('--output-dir', type=str, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(META_FILE)

    splits = get_patient_label(META_FILE)

    print(splits)

    for split_num in range(args.n_splits):
        output_fname = os.path.join(output_dir, f'split{split_num}.json')

        indices = splits[split_num]
        cn, ad, mci = [sum(df.iloc[indices]['label2'] == cat) for cat in ('CN', 'AD', 'MCI')]
        print(f'split {split_num}: {cn} cn {ad} ad {mci} mci {len(indices)} total')

        output = {
            'timestamp': str(datetime.now()),
            'number': split_num,
            'indices': indices,
            'metrics': {
                'cn': cn,
                'ad': ad,
                'mci': mci,
                'total': len(indices)
            }
        }

        with open(output_fname, 'w', encoding='utf-8') as output_file:
            json.dump(output, output_file, indent=4)

    print(f'done, splits are in {output_dir}')


if __name__ == '__main__':
    store_indices()

