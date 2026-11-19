import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from u_net import UNet
from pair_dataset import PairedOfflineDataset

def train_model():

    BATCH_SIZE = 32
    LEARNING_RATE = 1e-4
    EPOCHS = 100
    
    OLD_DIR = r"c:\Users\turtlepeer\project1\TA_project\final_demo_114b\celeba_hq_old\train"
    CLEAN_DIR = r"c:\Users\turtlepeer\project1\TA_project\final_demo_114b\celeba_hq\train"
    WEIGHTS_DIR = r"c:\Users\turtlepeer\project1\TA_project\final_demo_114b\weights"
    os.makedirs(WEIGHTS_DIR, exist_ok=True) 

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    if device.type != 'cuda':
        print("no gpu")

    dataset = PairedOfflineDataset(old_dir=OLD_DIR, clean_dir=CLEAN_DIR, image_size=(256, 256))
    
    dataloader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=8, 
        pin_memory=True,         # 將資料鎖定在分頁記憶體，加速轉移至 GPU
        persistent_workers=True  # 避免 Windows 每個 Epoch 重新啟動 Worker 的延遲
    )
    print(f"訓練資料集載入完成，共 {len(dataset)} 張圖片，分為 {len(dataloader)} 個 Batch。")

    model = UNet(in_channels=3, out_channels=3).to(device)
    
    criterion = nn.L1Loss() 
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scaler = torch.amp.GradScaler('cuda')


    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for batch_idx, (x_old, y_clean) in progress_bar:
            # 加入 non_blocking=True，讓 GPU 搬資料與 CPU 運算可以非同步進行
            x_old = x_old.to(device, non_blocking=True)
            y_clean = y_clean.to(device, non_blocking=True)
            
            optimizer.zero_grad() 
            
            # 使用 autocast 區塊包裝 Forward Pass，自動切換至 FP16 進行高速運算
            with torch.cuda.amp.autocast():
                outputs = model(x_old)
                loss = criterion(outputs, y_clean)
            
            # 使用 Scaler 進行 Backward Pass 與權重更新
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item()
            progress_bar.set_postfix({'L1_Loss': f"{loss.item():.4f}"})
            
        # 計算該 Epoch 的平均 Loss
        epoch_loss = running_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{EPOCHS}] Average Loss: {epoch_loss:.4f}\n")
        
        if (epoch + 1) % 5 == 0:
            save_path = os.path.join(WEIGHTS_DIR, f"unet_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"已自動儲存 Epoch {epoch+1} 的模型權重！")

    print("訓練全數完成！可以準備進行推論測試了。")

if __name__ == "__main__":
    train_model()