import re, os, glob, sys

sys.path.insert(0, '.')
os.chdir(r'E:\BAISYS_QUANT\BAISYS_QUANT')

files = glob.glob('BackTrading/**/*.py', recursive=True) + glob.glob('DataManager/**/*.py', recursive=True)
windows = []

for f in sorted(set(files)):
    if not os.path.isfile(f):
        continue
    text = open(f, encoding='utf-8', errors='ignore').read()
    for m in re.finditer(r'\.rolling\s*\(\s*([0-9]+)', text):
        windows.append((f, 'rolling', m.group(1)))
    for m in re.finditer(r'\.shift\s*\(\s*(-?[0-9]+)', text):
        windows.append((f, 'shift', m.group(1)))
    for m in re.finditer(r'MACD_PARAMS.*?\(([0-9]+)\s*,\s*([0-9]+)\s*,\s*([0-9]+)\)', text):
        windows.append((f, 'MACD_slow', m.group(2)))
    for m in re.finditer(r'RSI[_\s]*N\s*=\s*([0-9]+)', text):
        windows.append((f, 'RSI_N', m.group(1)))
    for m in re.finditer(r'(?:ATR|atr)\s*(?:_N|_WINDOW|period)?\s*=\s*([0-9]+)', text, re.I):
        windows.append((f, 'ATR', m.group(1)))
    for m in re.finditer(r'KDJ_N\s*=\s*([0-9]+)', text):
        windows.append((f, 'KDJ_N', m.group(1)))
    for m in re.finditer(r'MA([0-9]+)', text):
        windows.append((f, 'MA' + m.group(1), 'N/A'))
    for m in re.finditer(r'(?:window|max_window|min_periods)\s*=\s*([0-9]+)', text):
        windows.append((f, 'window', m.group(1)))

print('=== Indicators window analysis ===')
for entry in sorted(set(windows)):
    print(f'  {entry[0]}: {entry[1]:12s} = {entry[2]}')

# Also check config.ini for indicator periods
if os.path.exists('config.ini'):
    ini = open('config.ini', encoding='utf-8', errors='ignore').read()
    for m in re.finditer(r'MACD_PARAMS\s*=\s*([0-9]+)\s*,\s*([0-9]+)\s*,\s*([0-9]+)', ini):
        print(f'  config.ini: MACD_slow    = {m.group(2)}')
    for m in re.finditer(r'(?:RSI_N|KDJ_N|ATR_N|CCI_N|ROC_N)\s*=\s*([0-9]+)', ini):
        print(f'  config.ini: ind         = {m.group(1)}')
