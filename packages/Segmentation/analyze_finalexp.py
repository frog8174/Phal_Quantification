import pandas as pd
import sys
import os

sys.stdout.reconfigure(encoding='utf-8')

base = r'y:\Workspace\Aaron\2026-CIGR-phal-yolo-seg-quantify\packages\Segmentation\training_result'
runs = ['Finalexp_lastlayer_lr2e-4_v1', 'Finalexp_lastlayer_lr4e-4_v1', 'Finalexp_lastlayer_v1', 'baseline_unet']

for name in runs:
    log = os.path.join(base, name, 'log.csv')
    params = os.path.join(base, name, 'params.csv')
    if not os.path.exists(log):
        continue
    df = pd.read_csv(log)
    
    lr_h = lr_b = 'N/A'
    if os.path.exists(params):
        pdf = pd.read_csv(params)
        if 'lr_head' in pdf.columns:
            lr_h = pdf['lr_head'].iloc[0]
        if 'lr_backbone' in pdf.columns:
            lr_b = pdf['lr_backbone'].iloc[0]
        if 'lr' in pdf.columns and lr_h == 'N/A':
            lr_h = pdf['lr'].iloc[0]
    
    best_miou = df['val_miou'].max()
    best_epoch = df['val_miou'].idxmax() + 1
    
    last = df.iloc[-1]
    gap_last = last['val_loss'] - last['train_loss']
    
    tail = df.tail(10)
    train_trend = tail['train_loss'].iloc[-1] - tail['train_loss'].iloc[0]
    val_trend = tail['val_loss'].iloc[-1] - tail['val_loss'].iloc[0]
    miou_trend = tail['val_miou'].iloc[-1] - tail['val_miou'].iloc[0]
    
    print(f'=== {name} ===')
    print(f'  LR: head={lr_h}, backbone={lr_b}')
    print(f'  Total epochs: {len(df)}')
    print(f'  Best mIoU: {best_miou:.4f} (epoch {best_epoch})')
    
    tl = last['train_loss']
    vl = last['val_loss']
    mi = last['val_miou']
    print(f'  Final: T_Loss={tl:.4f}, V_Loss={vl:.4f}, mIoU={mi:.4f}')
    print(f'  Loss Gap (V-T): {gap_last:.4f}')
    print(f'  Last 10 ep trend: T_Loss {train_trend:+.4f}, V_Loss {val_trend:+.4f}, mIoU {miou_trend:+.4f}')
    
    if val_trend > 0 and train_trend < 0:
        print(f'  Status: Overfitting')
    elif val_trend < 0 and miou_trend > 0:
        print(f'  Status: Still improving')
    elif abs(val_trend) < 0.01 and abs(miou_trend) < 0.005:
        print(f'  Status: Converged')
    else:
        print(f'  Status: Plateau / Fluctuating')
    print()
