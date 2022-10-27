import os
import pandas as pd
from tqdm import tqdm

if __name__ == '__main__':

    df = pd.DataFrame()

    home_path = os.getcwd()

    os.chdir('images')

    images_path = os.getcwd()

    for folder in tqdm(os.listdir()):

        os.chdir(os.path.join(images_path, folder))

        files = os.listdir()

        curr_df = pd.DataFrame({'room_id' : files,
                                'simple' : [0 for _ in range(len(files))],
                                'complex' : [0 for _ in range(len(files))]})

        df = pd.concat([df, curr_df])

    os.chdir(home_path)
    os.chdir('data')

    df.to_csv('data.csv', index = False)

    print('Script ran!')



