# Prepare DailyDialog annotations.

import csv
import pandas as pd

act_df = pd.read_csv('dialogues_act.txt', header=None)
text_df = pd.read_csv('dialogues_text.txt', delimiter="\t", header=None)
acts = [l[0].split() for l in act_df.values.tolist()]
texts = [l[0].split('__eou__')[:-1] for l in text_df.values.tolist()]

assert len(acts) == len(texts)

speech_acts = []

for i in range(len(acts)):
    for j in range(len(acts[i])):
        if acts[i][j] == '1':
            speech_acts.append({'inform': texts[i][j]})
        if acts[i][j] == '2':
            speech_acts.append({'question': texts[i][j]})
        if acts[i][j] == '3':
            speech_acts.append({'directive': texts[i][j]})
        if acts[i][j] == '4':
            speech_acts.append({'commissive': texts[i][j]})

inform = []
question = []
directive = []
commissive = []

for d in speech_acts:
    if 'inform' in d:
        for v in d.values():
            inform.append(v)

for d in speech_acts:
    if 'question' in d:
        for v in d.values():
            question.append(v)

for d in speech_acts:
    if 'directive' in d:
        for v in d.values():
            directive.append(v)

for d in speech_acts:
    if 'commissive' in d:
        for v in d.values():
            commissive.append(v)

with open('inform.txt', 'a', encoding='utf-8') as inform_samples:
    inform_samples.write('Text\n')
    for s in inform:
        inform_samples.write(s + '\n')

with open('question.txt', 'a', encoding='utf-8') as question_samples:
    question_samples.write('Text\n')
    for s in question:
        question_samples.write(s + '\n')

with open('directive.txt', 'a', encoding='utf-8') as directive_samples:
    directive_samples.write('Text\n')
    for s in directive:
        directive_samples.write(s + '\n')

with open('commissive.txt', 'a', encoding='utf-8') as commissive_samples:
    commissive_samples.write('Text\n')
    for s in commissive:
        commissive_samples.write(s + '\n')

inform_df = pd.read_csv('inform.txt', sep='\t', error_bad_lines=False, engine='python')
question_df = pd.read_csv('question.txt', sep='\t', error_bad_lines=False, engine='python')
directive_df = pd.read_csv('directive.txt', sep='\t', error_bad_lines=False, engine='python')
commissive_df = pd.read_csv('commissive.txt', sep='\t', error_bad_lines=False, engine='python')

per_cat_max = len(commissive_df)

inform = inform_df['Text'].tolist()[:per_cat_max]
question = question_df['Text'].tolist()[:per_cat_max]
directive = directive_df['Text'].tolist()[:per_cat_max]
commissive = commissive_df['Text'].tolist()[:per_cat_max]

with open('equalized.csv', mode='w', encoding='utf-8') as csv_file:
    fieldnames = ['text', 'label']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()
    for sample in inform:
        writer.writerow({'text': sample, 'label': 1})
    for sample in question:
        writer.writerow({'text': sample, 'label': 2})
    for sample in directive:
        writer.writerow({'text': sample, 'label': 3})
    for sample in commissive:
        writer.writerow({'text': sample, 'label': 4})
