#!/usr/bin/python3

import sys
from collections import defaultdict
from operator import itemgetter

def dump(name, counts):
    items = sorted(counts.items()) # , key=itemgetter(1))
    for a, b in items:
        print('%-17s:    %-25s = %5d' % (name, a, b))
    counts.clear()         

counts = defaultdict(int)
name = ''
for line in sys.stdin:
    words = line.strip().split()
    # print(words)
    if len(words) >= 3 and words[0] == ';' and words[1][-1] == ':' and len(words[1]) <= 25 and not words[1].endswith('ForWavesPerEU:'):
        #in ('NumSgprs:', 'NumVgprs:', 'FloatMode', 'LDSByteSize', 'NumSGPRsForWavesPerEU:', 'NumVGPRsForWavesPerEU:', 'Occupancy:', 'LDSByteSize:', 'ScratchSize'):
        try:
            counts[words[1]] = int(words[2])
        except:
            pass
    elif len(words) >= 4 and words[1] == 'codeLenInByte':
        counts[words[1]] = int(words[3])
    elif len(words) >= 5 and words[-3:-1] == ['Begin', 'function']:
        if name:
            dump(name, counts)
        name = words[-1]
        
    word = words[0] if words else ''
    
#    if word == '.amdhsa_kernel':
#        name = words[1]
#    if (word == '.text' or word == '---') and counts and name:
#        dump(name, counts)
#        name = None
    if word and word[0] not in ';.' and word[-1] not in ':':
        counts[word] += 1

if name:
    dump(name, counts)
