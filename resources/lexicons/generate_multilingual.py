"""
This script is to convert the english gender-related tokens to the other languages
"""

source_file = 'replace_english.txt'
tokens = set()
with open(source_file) as dfile:
	for line in dfile:
		tokens.add(line.strip().lower())

flist = [
	['italian', 'it-en.txt'],
	['danish', 'da-en.txt'],
	['french', 'fr-en.txt'],
	['polish', 'pl-en.txt'],
	['spanish', 'es-en.txt'],
	['portuguese', 'pt-en.txt'],
	['german', 'de-en.txt']
]

for (lang, fpath) in flist:
	print(lang)
	lexicon = dict()
	with open(fpath) as dfile:
		for line in dfile:
			line = line.strip()
			if len(line) < 3:
				continue
			line = line.split()
			if line[1] in tokens:
				lexicon[line[1]] = line[0]
	with open('replace_{}.txt'.format(lang), 'w') as wfile:
		for token in tokens:
			wfile.write(lexicon.get(token, token)+'\n')
