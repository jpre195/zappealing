import pandas as pd
import argparse
import os
from tqdm import tqdm
import numpy as np
import shutil

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--simple', action = 'store_true')

def prep_folder(final_folder: str):

    if not os.path.exists(f'data/images/{final_folder}'):

        os.makedirs(f'data/images/{final_folder}')

def train_test_label(df):

    label = np.random.binomial(size = 1, n = 1, p = .2)

    if label == 0:

        df['group'] = 'train'

    else:

        df['group'] = 'test'

    return df

if __name__ == '__main__':

    args = parser.parse_args()

    print('Reading in data...', end = '')
    df = pd.read_csv(f'data/data_labeled.csv')
    print('Complete')

    if args.simple:

        prep_folder('train/like')
        prep_folder('train/dislike')
        prep_folder('test/like')
        prep_folder('test/dislike')

        room_dict = {'bath' : 'Bathroom',
                    'bed' : 'Bedroom',
                    'din' : 'Dinning',
                    'kitchen' : 'Kitchen',
                    'living' : 'Livingroom'}

        label_dict = {0 : 'dislike',
                    1 : 'like'}

        df = df.apply(train_test_label, axis = 1)

        print(df.value_counts('group'))

        for room, group, label, path in tqdm(zip(df['room'], df['group'], df['simple'], df['room_id'])):

            destination = f'data/images/{group}/{label_dict[label]}/{path}'

            room_path = f'images/{room_dict[room]}/{path}'

            shutil.copy(room_path, destination)


            

