"""Preprocessing steps"""
import numpy as np
from dateutil.parser import parse
import os
import ast
from tqdm import tqdm


def extract_trustpilot(dpath, opath, utable_path):
    utable = dict()

    with open(opath, 'w') as wfile:
        wfile.write('\t'.join(
            ['did', 'uid', 'text', 'date', 'gender', 'age', 'country', 'label']
        ) + '\n')
        with open(dpath) as dfile:
            for line in dfile:
                # line = json.loads(line)
                line = ast.literal_eval(line)
                uid = str(line['user_id']).strip()
                if len(uid) < 2:
                    continue
                if uid not in utable:
                    utable[uid] = dict()
                    utable[uid]['age'] = []

                gender = line.get('gender', '')
                if gender is None:
                    gender = 'x'
                gender = gender.strip().lower()
                if len(gender) == 0:
                    gender = 'x'
                utable[uid]['gender'] = gender

                age = line.get('birth_year', 'x')
                if age is None:
                    age = 'x'

                if 'country' in line:
                    country = line['country']
                    if country is None:
                        country = 'x'
                    else:
                        country = country.strip().split('_')
                        country = [word.capitalize() for word in country]
                        country = ' '.join(country)
                else:
                    country = line.get('location', 'x')
                    if country is None:
                        country = 'x'
                    else:
                        country = country.strip()
                        if len(country) == 0:
                            country = 'x'
                        else:
                            country = country.split()[-1]
                utable[uid]['country'] = country

                for idx, review in enumerate(line['reviews']):
                    date = review.get('date', None)
                    if date is None:
                        continue
                    date = parse(date)
                    if age != 'x':
                        age = int(age)
                        cur_age = date.year - age
                        utable[uid]['age'].append(cur_age)
                        cur_age = str(cur_age)
                    else:
                        cur_age = 'x'
                    date = date.strftime('%Y-%m-%d')

                    text = ''.join(review['text'])
                    text = text.strip().replace('\t', '').replace('\n', '')
                    did = '{}_{}'.format(review['company_id'], idx)
                    label = str(review['rating'])

                    wfile.write('\t'.join([
                        did, uid, text, date, gender, cur_age, country, label
                    ]) + '\n')

                utable[uid]['age'] = [int(item) for item in utable[uid]['age'] if item != 'x']
                if len(utable[uid]['age']) == 0:
                    utable[uid]['age'] = 'x'
                else:
                    utable[uid]['age'] = str(np.mean(utable[uid]['age']))

    with open(utable_path, 'w') as wfile:
        wfile.write('uid\tgender\tage\tcountry\n')
        for uid in utable:
            wfile.write('\t'.join(
                [uid, utable[uid]['gender'], utable[uid]['age'], utable[uid]['country']]
            ) + '\n')


data_dir = './review/trustpilot/'
for dpath in tqdm([fpath for fpath in os.listdir(data_dir) if fpath != '.DS_Store']):
    dname = dpath.split('.')[0]
    extract_trustpilot(
        data_dir + dpath, '{}{}.tsv'.format(data_dir, dname), '{}/{}_utable.tsv'.format(data_dir, dname)
    )
