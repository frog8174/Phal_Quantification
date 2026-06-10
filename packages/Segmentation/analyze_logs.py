import os
import pandas as pd

base_dir = r'y:\Workspace\Aaron\2026-CIGR-phal-yolo-seg-quantify\packages\Segmentation\training_result'
subdirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

results = []

for d in subdirs:
    dir_path = os.path.join(base_dir, d)
    log_path = os.path.join(dir_path, 'log.csv')
    params_path = os.path.join(dir_path, 'params.csv')
    
    if not os.path.exists(log_path):
        continue
        
    try:
        df = pd.read_csv(log_path)
        if df.empty or 'val_miou' not in df.columns or 'val_loss' not in df.columns:
            continue
            
        lr_head = df['lr_head'].iloc[0] if 'lr_head' in df.columns else 'N/A'
        lr_bb = df['lr_backbone'].iloc[0] if 'lr_backbone' in df.columns else 'N/A'
        
        layer_info = d
        if os.path.exists(params_path):
            try:
                pdf = pd.read_csv(params_path)
                if 'extract_layers' in pdf.columns:
                    layer_info = str(pdf['extract_layers'].iloc[0])
                if lr_head == 'N/A' and 'lr_head' in pdf.columns:
                    lr_head = pdf['lr_head'].iloc[0]
                if lr_bb == 'N/A' and 'lr_backbone' in pdf.columns:
                    lr_bb = pdf['lr_backbone'].iloc[0]
            except:
                pass
                
        max_miou = df['val_miou'].max()
        best_epoch_miou = df['val_miou'].idxmax() + 1
        
        min_val_loss = df['val_loss'].min()
        best_epoch_loss = df['val_loss'].idxmin() + 1
        
        # 檢查最後 5 個 epoch 的 loss 趨勢
        last_epochs = df.tail(5)
        if len(last_epochs) >= 2:
            val_loss_trend = last_epochs['val_loss'].iloc[-1] - last_epochs['val_loss'].iloc[0]
            train_loss_trend = last_epochs['train_loss'].iloc[-1] - last_epochs['train_loss'].iloc[0]
        else:
            val_loss_trend = 0
            train_loss_trend = 0
            
        # 狀態判定
        status = 'Unknown'
        if val_loss_trend > 0 and train_loss_trend < 0:
            status = 'Overfitting (Val Loss 上升但 Train Loss 下降)'
        elif val_loss_trend < 0:
            status = 'Underfitting / 仍有進步空間 (Val Loss 持續下降中)'
        else:
            status = 'Converged (收斂 / 平穩)'
            
        results.append({
            'Run': d,
            'Layer': layer_info,
            'LR(H/B)': f'{lr_head}/{lr_bb}',
            'Max_mIoU': round(max_miou, 4),
            'mIoU_Epoch': best_epoch_miou,
            'Min_Val_Loss': round(min_val_loss, 4),
            'Status': status,
            'Total_Epochs': len(df)
        })
            
    except Exception as e:
        pass

# 依 mIoU 排序
results = sorted(results, key=lambda x: x['Max_mIoU'], reverse=True)

# 顯示前幾名
print('--- Training Results Comparison ---')
for idx, r in enumerate(results):
    print(f"{idx+1}. {r['Run']}")
    print(f"   Layers: {r['Layer']} | LRs: {r['LR(H/B)']}")
    print(f"   Max mIoU: {r['Max_mIoU']} (at epoch {r['mIoU_Epoch']}/{r['Total_Epochs']})")
    print(f"   Status: {r['Status']}")
    print('-' * 40)
