import ast
import sys

with open('BackTrading/engine/core.py', encoding='utf-8') as f:
    source = f.read()
try:
    ast.parse(source)
    print('AST parse OK, no syntax errors')
except SyntaxError as e:
    print(f'SYNTAX ERROR at line {e.lineno}: {e.msg}')
    lines = source.splitlines()
    if e.lineno:
        start = max(0, e.lineno - 3)
        end = min(len(lines), e.lineno + 2)
        for i in range(start, end):
            marker = '>>>' if i + 1 == e.lineno else '   '
            print(f'{marker} {i+1}: {lines[i]}')
    sys.exit(1)
