#!/usr/bin/python3

import sys
from collections import defaultdict
from operator import itemgetter

def dump(name, counts):
    items = sorted(counts.items()) # , key=itemgetter(1))
    for a, b in items:
        print('%-17s:    %-25s %5d' % (name, a, b))
    counts.clear()         

counts = defaultdict(int)
name = ''
for line in sys.stdin:
    words = line.strip().split()
    # print(words)
    if len(words) >= 3 and words[1] in ('NumSGPRsForWavesPerEU:', 'NumVGPRsForWavesPerEU:', 'Occupancy:', 'LDSByteSize:'):
        counts[words[1]] = int(words[2])
    elif len(words) >= 4 and words[1] == 'codeLenInByte':
        counts[words[1]] = int(words[3])
    
    word = words[0] if words else ''
    
    if word == '.amdhsa_kernel':
        name = words[1]
    elif (word == '.text' or word == '---') and counts:
        dump(name, counts)
    elif word and word[0] not in ';.' and word[-1] not in ':':
        counts[word] += 1

# dump(name, counts)
