#!/usr/bin/python3

import sys

lineNo = 0
current = None
body = None
macros = {}

def err(text):
    print(f'#{lineNo}', text, file=sys.stderr)
    exit(1)

for line in sys.stdin:
    lineNo += 1

    isBegin = line.startswith('//{{')
    isEnd   = line.startswith('//}}')
    isInline = line.startswith('//==')
    name = line[4:].strip()
    if isBegin:
        if current:
            err(f'starting {name} while {current} is active')
        elif name in macros:
            err(f'{name} already defined')
        else:
            current = name
            body = ''    
    elif isEnd:
        if name != current:
            err(f'closing {name} while {current} is active')
        else:
            macros[current] = body
            current = None
    else:
        if isInline:
            if name in macros:
                line = macros[name]
            else:
                err(f'macro {name} not found')

        if current:
            body += line
        else:
            print(line,end='')
            
