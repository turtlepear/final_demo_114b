import os
import glob
import cv2
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class PairedOfflineDataset(Dataset):
    """讀取已經處理好的實體 (老照片, 乾淨原圖) 資料夾"""
    def __init__(self, old_dir, clean_dir, categories=['female', 'male'], image_size=(256, 256)):
        self.old_dir = old_dir
        self.clean_dir = clean_dir
        self.image_size = image_size
        self.image_filenames = []
        
        # 遍歷類別資料夾，收集相對路徑 (例如: 'female/000001.jpg')
        for category in categories:
            clean_category_dir = os.path.join(clean_dir, category)
            for img_path in glob.glob(os.path.join(clean_category_dir, '*.jpg')):
                filename = os.path.basename(img_path)
                self.image_filenames.append(os.path.join(category, filename))
                
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        rel_path = self.image_filenames[idx]
        
        # 組合出兩邊的絕對路徑
        old_path = os.path.join(self.old_dir, rel_path)
        clean_path = os.path.join(self.clean_dir, rel_path)
        
        # 讀取圖片
        old_img = cv2.imread(old_path)
        clean_img = cv2.imread(clean_path)
        
        # 調整大小與色彩空間
        old_img = cv2.cvtColor(cv2.resize(old_img, self.image_size), cv2.COLOR_BGR2RGB)
        clean_img = cv2.cvtColor(cv2.resize(clean_img, self.image_size), cv2.COLOR_BGR2RGB)
        
        return self.transform(old_img), self.transform(clean_img)