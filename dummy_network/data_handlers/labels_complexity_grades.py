import pandas as pd

path = '/Users/vojtechremis/Desktop/Projects/birdsong-segmentation/dummy_network/data_handlers/labels/labels.csv'
df = pd.read_csv(path, delimiter=',')


def complexity(motifs):
    if motifs < 5:
        return 4
    if motifs < 10:
        return 3
    if motifs < 20:
        return 2
    if motifs < 30:
        return 1
    if motifs < 40:
        return 0
    if motifs < 50:
        return -1

df['complexity'] = df['motifs'].apply(complexity)

df.to_csv(path, index=False)



