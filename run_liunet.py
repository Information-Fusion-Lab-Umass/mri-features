#!/usr/bin/env python3

import argparse
from datetime import datetime
import dateutil
import json
import os
import re
import sys
import traceback

import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
import torch
from torch import optim
from torch import autograd
from torch.utils.data import DataLoader, Subset, SubsetRandomSampler
from tqdm import tqdm

from adni_dataset import AdniDataset
from liunet import LiuNet

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
DEFAULT_CHECKPOINT_DIR = os.path.join(SCRIPT_DIR, 'checkpoints')
DEFAULT_OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'output')
DEFAULT_SPLIT_DIR = os.path.join(SCRIPT_DIR, 'splits')
META_FILE = os.path.join(SCRIPT_DIR, 'mriimg_meta_v4.csv')

class Tee:
    def __init__(self, name, mode):
        self.file = open(name, mode)
        self.stdout = sys.stdout
        sys.stdout = self
    def __enter__(self):
        pass
    def __exit__(self, etype, value, tb):
        if tb is not None:
            traceback.print_exception(etype, value, tb, file=self.file)
        sys.stdout = self.stdout
        self.file.close()
    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)
    def flush(self):
        self.file.flush()

def default_log_file(split_num):
    basename = datetime.now().strftime('%Y%m%d_%H%M%S') + f'_split{split_num}.log'
    return os.path.join(SCRIPT_DIR, 'logs', basename)

def load_splits(split_dir, split_nums, limit=None, check_timestamp=None):
    splits = {}
    for split_num in split_nums:
        split_fname = os.path.join(split_dir, f'split{split_num}.json')
        with open(split_fname, 'r', encoding='utf-8') as split_file:
            split_obj = json.load(split_file)
        assert split_obj['number'] == split_num
        split = sorted(split_obj['indices'])
        timestamp = dateutil.parser.parse(split_obj['timestamp'])

        if limit is not None:
            split = split[:limit]

        if check_timestamp is not None:
            assert np.abs((timestamp - check_timestamp).total_seconds()) < 5, f"Split {split_num} was generated at a different time than the input split"

        splits[split_num] = (split, timestamp)

    return splits

def train(model, train_loader, optimizer, device, epoch, name, debug_mode):
    model.train()

    losses = []

    with autograd.set_detect_anomaly(debug_mode):
        for batch_idx, (_, input, target) in tqdm(enumerate(train_loader), total=len(train_loader)):
            input, target = input.to(device), target.to(device)
            optimizer.zero_grad()
            preds = model(input)
            loss = model.criterion(preds, target)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            
            if batch_idx % 100 == 0:
                print(f"[{name}] epoch {epoch}\tbatch {batch_idx}\tloss {loss}\tmean {np.mean(losses)}")

    print(f"[{name}] epoch {epoch}\tmean loss {np.mean(losses)}")

def test(model, test_loader, device, name, debug_mode):
    model.eval()

    all_feats = []
    all_preds = []
    losses = []
    ordering = []

    with autograd.set_detect_anomaly(debug_mode):
        with torch.no_grad():
            for indices, input, target in tqdm(test_loader, total=len(test_loader)):
                input, target = input.to(device), target.to(device)
                feats = model.encode(input)
                preds = model.classifier(feats)
                loss = model.criterion(preds, target)

                all_feats.extend(feats)
                all_preds.extend(preds)
                losses.append(loss.item())
                ordering.extend(indices)

    print(f"[{name}] mean loss {np.mean(losses)}")
    
    # SubsetRandomSampler doesn't sample sequentially. We have to rearrange the predictions in sequential order
    # so that we can compare them to the appropriate labels
    all_feats = [feats for _, feats in sorted(zip(ordering, all_feats), key=lambda t: t[0])]
    all_preds = [preds for _, preds in sorted(zip(ordering, all_preds), key=lambda t: t[0])]

    return torch.stack(all_feats, dim=0), torch.stack(all_preds, dim=0), np.mean(losses)

def best_model_path(checkpoint_dir, split_num):
    os.makedirs(checkpoint_dir, exist_ok=True)
    return os.path.join(os.path.abspath(checkpoint_dir), f'best_model_split{split_num}.pt')

def output_path(input_path, output_dir, split_num):
    output_dir = os.path.abspath(output_dir)
    fname = os.path.basename(input_path).replace('.nii.gz', '')
    parts = fname.split('_')
    patient_id = parts[0].replace('sub-ADNI', '')
    assert re.match(r'\d{3}S\d{4}', patient_id)
    viscode = parts[1].replace('ses-', '').lower()
    visit_output_dir = os.path.join(output_dir, f'split{split_num}', patient_id, viscode)

    os.makedirs(visit_output_dir, exist_ok=True)
    return os.path.join(visit_output_dir, 'features.pt')

def main():
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    #torch.set_deterministic(True)
    #os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--checkpoint-dir', type=str, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--learning-rate', type=float, default=0.01)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--log-file', type=str)
    parser.add_argument('--n-epochs', type=int, default=30)
    parser.add_argument('--n-splits', type=int, default=10)
    parser.add_argument('--output-dir', type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--split', type=int, required=True)
    parser.add_argument('--split-dir', type=str, default=DEFAULT_SPLIT_DIR)
    args = parser.parse_args()

    split_num = args.split
    log_file = args.log_file or default_log_file(split_num)
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    with Tee(log_file, 'w'):

        debug_mode = args.debug
        print(f"debug mode is {'ON' if debug_mode else 'OFF'}")

        # Model setup
        assert torch.cuda.is_available(), "Make sure you're running this script on GYPSUM, and not on the head node"
        device = torch.device('cuda:0')
        model = LiuNet().to(device)
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)

        # Data setup

        df = pd.read_csv(META_FILE)
        m, n = df.shape
        dataset = AdniDataset(df)

        split, timestamp = load_splits(args.split_dir, [split_num], limit=args.limit)[split_num]
        print(f"loaded split {split_num} with {len(split)} images")

        # The split passed via --split is the test set.
        # We train/validate on the other splits, using the resultant model to generate features for the test set.
        n_splits = args.n_splits
        val_split_num = (split_num + 1) % n_splits

        other_split_nums = list(range(0, split_num)) + list(range(split_num+1, n_splits))
        train_split_nums = [num for num in other_split_nums if num != val_split_num]
        other_splits = load_splits(args.split_dir, other_split_nums, limit=args.limit, check_timestamp=timestamp)
        
        train_idx = sorted(np.concatenate([other_splits[num][0] for num in train_split_nums], axis=0))
        val_idx = other_splits[val_split_num][0]
        test_idx = split

        print(f"training on splits {{{' '.join([str(num) for num in train_split_nums])}}} with {len(train_idx)} images")
        print(f"using val split {val_split_num} with {len(val_idx)} images")

        for indices in (train_idx, val_idx, test_idx):
            assert len(indices) == len(set(indices))
            assert indices == sorted(indices)
        all_idx = train_idx+val_idx+test_idx
        assert len(all_idx) == len(set(all_idx))

        '''
        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)
        test_sampler = SubsetRandomSampler(test_idx)

        train_loader = DataLoader(dataset, sampler=train_sampler, batch_size=args.batch_size)
        val_loader = DataLoader(dataset, sampler=val_sampler, batch_size=args.batch_size)
        test_loader = DataLoader(dataset, sampler=test_sampler, batch_size=args.batch_size)
        '''

        # TODO: Revert to random sampling once we figure out why we're getting nans
        train_dataset = Subset(dataset, train_idx)
        val_dataset = Subset(dataset, val_idx)
        test_dataset = Subset(dataset, test_idx)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

        print()

        # Model training
        print("starting training")
        best_epoch, best_loss = -1, np.inf

        for epoch in range(args.n_epochs):
            print(f"epoch {epoch}/{args.n_epochs}")
            train(model, train_loader, optimizer, device, epoch, 'train', debug_mode)
            val_loss = test(model, val_loader, device, 'val', debug_mode)[-1]
            
            if val_loss < best_loss:
                torch.save(model.state_dict(), best_model_path(args.checkpoint_dir, split_num))
                best_epoch, best_loss = epoch, val_loss

        print("finished training")
        print(f"got best model at epoch {best_epoch} with val loss {best_loss}")
        print()
        
        # Model evaluation and feature generation
        print("generating feats")
        checkpoint = torch.load(best_model_path(args.checkpoint_dir, split_num))
        model.load_state_dict(checkpoint)
        test_feats, test_probs, _ = test(model, test_loader, device, 'test', debug_mode)

        # Output the metrics for our test predictions to make sure that the model is performing well
        test_labels = dataset.labels[test_idx]
        test_probs = softmax(test_probs.cpu().numpy(), axis=1)
        test_preds = np.argmax(test_probs, axis=1)
        print(metrics.classification_report(test_labels, test_preds, labels=range(3), target_names=dataset.categories))
        
        all_classes_present = all([(label in test_labels) for label in range(3)])
        if all_classes_present:
            print("roc/auc (macro, ovr):", metrics.roc_auc_score(test_labels, test_probs, average='macro', multi_class='ovr'))
            print("roc/auc (micro, ovr):", metrics.roc_auc_score(test_labels, test_probs, average='micro', multi_class='ovr'))
            print("roc/auc (macro, ovo):", metrics.roc_auc_score(test_labels, test_probs, average='macro', multi_class='ovo'))
            print("roc/auc (micro, ovo):", metrics.roc_auc_score(test_labels, test_probs, average='micro', multi_class='ovo'))

        conf_matrix = metrics.confusion_matrix(test_labels, test_preds, labels=range(3))
        print("confusion matrix:")
        print(conf_matrix)
        print("labels:", ' '.join(dataset.categories))
        print()

        # Save the each of the image features to a file
        input_paths = list(df['caps_path'].iloc[test_idx])
        output_paths = [output_path(path, args.output_dir, split_num) for path in input_paths]

        for img_feats, path in zip(test_feats, output_paths):
            torch.save(img_feats, path)
        
        print("finished generating feats")
        print()

if __name__ == '__main__':
    main()
