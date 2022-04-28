#!/usr/bin/python3

import sys

# If any arguments are passed, expect <input-file> <output-file>
# Otherwise read from stdin, write to stdout
if len(sys.argv) > 1:
    assert len(sys.argv) == 3, f'Use: f{sys.argv[0]} <input-file> <output-file>'
    sys.stdin  = open(sys.argv[1])
    sys.stdout = open(sys.argv[2], 'w')

HEAD = 'const char *CL_SOURCE = R"clsource('
TAIL = ')clsource";';
    
lineNo = 0
current = None
body = None
macros = {}

print(HEAD)

def err(text):
    print(f'#{lineNo}', text, file=sys.stderr)
    exit(1)

for line in sys.stdin:
    lineNo += 1
    line = line.lstrip()
    
    if line.startswith('//{{ '):
        name = line[5:].strip()
        if current:
            err(f'starting template {name} while {current} is active')
        else:
            current = name
            body = ''
    elif line.startswith('//}}'):
        if not current:
            err(f'template end without begin')
        else:
            macros[current] = body
            current = None
    else:
        if line.startswith('//== '):
            name, _, tail = line[5:].partition(' ')
            if name not in macros:
                err(f'template {name} not defined')

            body = macros[name]
            args = map(str.strip, tail.split(','))
            for arg in args:
                key, value = map(str.strip, arg.split('='))
                body = body.replace(key, value)
                line = body        

        if current:
            body += line
        else:
            print(line, end='')

print(TAIL)

