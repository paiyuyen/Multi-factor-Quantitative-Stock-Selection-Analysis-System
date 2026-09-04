import ast

with open(r'E:\BAISYS_QUANT\BAISYS_QUANT\BackTrading\engine\core.py', encoding='utf-8') as f:
    source = f.read()

tree = ast.parse(source)

for node in ast.iter_child_nodes(tree):
    if isinstance(node, ast.FunctionDef):
        nested = []
        for child in ast.walk(node):
            if isinstance(child, ast.FunctionDef) and child is not node:
                nested.append(f'    {child.name} (L{child.lineno})')
        if nested:
            print(f'{node.name} (L{node.lineno}):')
            for n in nested:
                print(n)
            print()

# Also list all top-level functions
print('=== Top-level functions ===')
for node in ast.iter_child_nodes(tree):
    if isinstance(node, ast.FunctionDef):
        print(f'  {node.name} (L{node.lineno})')

# Find nonlocal uses
print('\n=== nonlocal uses ===')
for node in ast.walk(tree):
    if isinstance(node, ast.Nonlocal):
        # find enclosing function line
        print(f'  L{node.lineno}: {", ".join(node.names)}')
