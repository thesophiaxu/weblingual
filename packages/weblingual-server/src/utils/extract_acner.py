import sys
import json

loc = sys.argv[1]
loc_target = sys.argv[2]
file = open(loc, encoding="windows-1252")
sentences = []
tokens = []
labels = []
for line in file:
    things = line.strip().split('\t')
    if len(things) == 3:
        sentences.append({'tokens': tokens, 'labels': labels})
        things[1] = things[1].replace('\"', '', 3)
        tokens = [things[1]]
        labels = [things[2]]
    else:
        things[0] = things[0].replace('\"', '', 3)
        tokens.append(things[0])
        labels.append(things[1])

sentences = sentences[1:]

outfile = open(loc_target, "w")
outfile.writelines([json.dumps(ln)+"\n" for ln in sentences])