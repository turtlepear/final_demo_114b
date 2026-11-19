import os
import glob
import cv2
from new_to_old import NewToOld

def process_and_save_dataset(base_in_dir, base_out_dir, categories):
    degrader = NewToOld(
        sepia_prob=0.9, 
        physical_damage_prob=0.85, 
        extreme_blur_prob=0.8, 
        severe_color_fade_prob=0.9
    )
    
    total_processed = 0

    for category in categories:
        in_dir = os.path.join(base_in_dir, category)
        out_dir = os.path.join(base_out_dir, category)
        
        os.makedirs(out_dir, exist_ok=True)
        
        image_paths = []
        for ext in ('*.jpg', '*.jpeg', '*.png'):
            image_paths.extend(glob.glob(os.path.join(in_dir, ext)))
            
        print(f"正在處理 [{category}] 類別，共找到 {len(image_paths)} 張圖片...")
        
        for i, img_path in enumerate(image_paths):
            clean_img = cv2.imread(img_path)
            if clean_img is None:
                print(f"警告：無法讀取 {img_path}，跳過此圖。")
                continue
                
            old_img = degrader(clean_img)
            
            filename = os.path.basename(img_path)
            save_path = os.path.join(out_dir, filename)
            
            cv2.imwrite(save_path, old_img, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            total_processed += 1
            
            if (i + 1) % 500 == 0:
                print(f"  - 已處理 {i + 1} / {len(image_paths)} 張")

    print(f"\n處理完畢！總共生成了 {total_processed} 張極度腐化的老照片。")
    print(f"請前往 {base_out_dir} 檢查您的新資料集！")

if __name__ == "__main__":
    BASE_IN_DIR = r"c:\Users\turtlepeer\project1\TA_project\final_demo_114b\celeba_hq\train"
    BASE_OUT_DIR = r"c:\Users\turtlepeer\project1\TA_project\final_demo_114b\celeba_hq_old\train"
    CATEGORIES = ['female', 'male']
    
    process_and_save_dataset(BASE_IN_DIR, BASE_OUT_DIR, CATEGORIES)