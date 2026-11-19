import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from u_net import UNet

def test_single_image(image_path, weight_path):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"正在使用 {device} 進行推論...")
    
    model = UNet(in_channels=3, out_channels=3).to(device)

    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"找不到權重檔：{weight_path}")
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()

    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError("無法讀取測試圖片！")
        

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (256, 256))
    
    img_normalized = img_resized.astype(np.float32) / 255.0
    
    input_tensor = torch.from_numpy(img_normalized).permute(2, 0, 1)
    
    input_batch = input_tensor.unsqueeze(0).to(device)

    print("正在修復圖片...")
    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            output_batch = model(input_batch)

    output_tensor = output_batch.squeeze(0).cpu()
    output_img = (output_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    axes[0].imshow(img_resized)
    axes[0].set_title("Input (Degraded Photo)")
    axes[0].axis('off')
    
    axes[1].imshow(output_img)
    axes[1].set_title("Output (Restored by U-Net)")
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    
    TEST_IMAGE_PATH = r"C:\Users\turtlepeer\project1\TA_project\final_demo_114b\restore-repair-fix-old-damaged-photo-colorize-image.jpg"
    WEIGHT_PATH = r"c:\Users\turtlepeer\project1\TA_project\final_demo_114b\weights\unet_epoch_70.pth"
    
    test_single_image(TEST_IMAGE_PATH, WEIGHT_PATH)