"""
Diagnose prediction quality for tatqa, docfinqa, multidoc2025.
Usage: python scripts/diagnose_predictions.py
"""
import json
import glob
import os

DATASETS = ['tatqa', 'docfinqa', 'multidoc2025']
OUTPUTS  = './outputs'
N_SHOW   = 5

for ds in DATASETS:
    # predictions file
    pred_files = sorted(glob.glob(os.path.join(OUTPUTS, f'{ds}_test_predictions_*.json')))
    if not pred_files:
        print(f'\n=== {ds}: no predictions file found ===')
        continue
    preds = json.load(open(pred_files[-1], encoding='utf-8'))

    # raw dataset file for context inspection
    if ds == 'multidoc2025':
        raw_path = './data/multidoc2025/test.json'
    else:
        raw_path = f'./data/benchmarks/{ds}/test.json'
    raw = json.load(open(raw_path, encoding='utf-8')) if os.path.exists(raw_path) else []

    print(f'\n{"="*60}')
    print(f'Dataset: {ds}  ({len(preds)} predictions, showing first {N_SHOW})')
    print(f'{"="*60}')

    # intent distribution
    from collections import Counter
    intents = Counter(s.get('intent','?') for s in raw[:200])
    print(f'Intent distribution (first 200): {dict(intents)}')

    # answer format distribution
    gt_numeric = sum(1 for s in raw[:200]
                     if str(s.get('answer','')).replace('.','').replace('-','').replace(',','').strip().isdigit()
                     or (str(s.get('answer','')).strip().startswith('$')))
    print(f'Numeric-looking GT (first 200): {gt_numeric}/200')

    print()
    for i, s in enumerate(preds[:N_SHOW]):
        q    = s.get('question', '')[:90]
        gt   = s.get('ground_truth', '')
        pred = s.get('prediction', '')

        # context length from raw dataset
        ctx_len = 0
        if i < len(raw):
            ctx = raw[i].get('context', '')
            if isinstance(ctx, list):
                ctx_len = sum(len(str(x)) for x in ctx)
            else:
                ctx_len = len(str(ctx))

        print(f'  [{i+1}] Q:    {q}')
        print(f'       GT:   {repr(str(gt)[:100])}')
        print(f'       Pred: {repr(str(pred)[:150])}')
        print(f'       ctx_len={ctx_len}')
        print()
