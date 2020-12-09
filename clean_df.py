#!/usr/bin/env python3

import pandas as pd

def label2(label):
    if label in ('LMCI', 'EMCI', 'SMC'):
        return 'MCI'
    return label

def main():
    cols = 'PTID,caps_date,raw_date,caps_path,bids_path,viscode,label,batch,date_delta'.split(',')
    df = pd.read_csv('mriimg_meta_v4.csv', usecols=cols)
    df['label2'] = df['label'].apply(label2)
    df.to_csv('mriimg_meta_v4.csv', index=False)

if __name__ == '__main__':
    main()
