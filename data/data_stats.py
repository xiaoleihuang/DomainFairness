"""Data stats"""

review_dir = './review/'
hate_speech_dir = './hatespeech/'

data_list = [
    ['review_trustpilot_english', review_dir + 'trustpilot/united_states.tsv', 'english'],
    ['review_trustpilot_french', review_dir + 'trustpilot/france.tsv', 'french'],
    ['review_trustpilot_german', review_dir + 'trustpilot/german.tsv', 'german'],
    ['review_trustpilot_danish', review_dir + 'trustpilot/denmark.tsv', 'danish'],
    ['hatespeech_twitter_english', hate_speech_dir + 'english/corpus.tsv', 'english'],
    ['hatespeech_twitter_spanish', hate_speech_dir + 'spanish/corpus.tsv', 'spanish'],
    ['hatespeech_twitter_italian', hate_speech_dir + 'italian/corpus.tsv', 'italian'],
    ['hatespeech_twitter_portuguese', hate_speech_dir + 'portuguese/corpus.tsv', 'portuguese'],
]

for item in data_list:
    data_stat = {
        'num_user': set(),
        'f-ratio': set(),
        'num_docs': 0,
        'avg_token': 0,
        'l-ratio': 0,
    }
    with open(item[1]) as dfile:
        cols = dfile.readline().strip().split('\t')
        uidx = cols.index('uid')
        gidx = cols.index('gender')
        tidx = cols.index('text')
        lidx = cols.index('label')
        for line in dfile:
            line = line.strip().lower().split('\t')
            if len(line) != len(cols):
                continue
            if len(line[tidx].strip().split()) < 10:
                continue

            # print(idx, line)
            if line[gidx] == 'x':
                continue

            data_stat['num_user'].add(line[uidx])
            data_stat['avg_token'] += len(line[tidx].split())

            # binarize labels in the trustpilot dataset to keep the same format.
            try:
                label = int(line[lidx])
            except ValueError:
                # encode hate speech data
                if line[lidx] in ['0', 'no', 'neither', 'normal']:
                    label = 0
                else:
                    label = 1

            # label trustpilot review scores
            if 'trustpilot' in item[1]:
                if label == 3:
                    continue
                elif label > 3:
                    label = 1
                else:
                    label = 0

            if label == 1:
                data_stat['l-ratio'] += 1
            if 'f' in line[gidx]:
                data_stat['f-ratio'].add(line[uidx])

            data_stat['num_docs'] += 1

    data_stat['l-ratio'] = data_stat['l-ratio'] / data_stat['num_docs']
    data_stat['avg_token'] = data_stat['avg_token'] / data_stat['num_docs']
    data_stat['num_user'] = len(data_stat['num_user'])
    data_stat['f-ratio'] = len(data_stat['f-ratio']) / data_stat['num_user']
    print('{} Data Stats:'.format(item[0]))
    print(data_stat)
    print()
