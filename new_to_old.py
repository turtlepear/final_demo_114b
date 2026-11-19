import cv2
import numpy as np
import random
from scipy.ndimage import gaussian_filter

class NewToOld:
    def __init__(self, sepia_prob=0.9, physical_damage_prob=0.8, 
                 extreme_blur_prob=0.7, severe_color_fade_prob=0.8):

        self.sepia_prob = sepia_prob
        self.physical_damage_prob = physical_damage_prob
        self.extreme_blur_prob = extreme_blur_prob
        self.severe_color_fade_prob = severe_color_fade_prob

    def __call__(self, image: np.ndarray) -> np.ndarray:
        old_img = image.copy()

        # 1. 嚴重色調退化 (大幅降低對比)
        if random.random() < self.severe_color_fade_prob:
            old_img = self._apply_severe_color_fade(old_img)

        # 2. 泛黃色調
        if random.random() < self.sepia_prob:
            old_img = self._apply_sepia(old_img)

        # 3. 物理刮痕
        if random.random() < self.physical_damage_prob:
            old_img = self._add_heavy_physical_damage(old_img)

        # 4. 模糊
        if random.random() < self.extreme_blur_prob:
            old_img = self._add_extreme_blur(old_img)

        return old_img

    def _apply_severe_color_fade(self, img: np.ndarray) -> np.ndarray:
        alpha = random.uniform(0.4, 0.6)
        beta = random.randint(30, 60)
        return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    def _apply_sepia(self, img: np.ndarray) -> np.ndarray:
        img_float = img.astype(np.float64)
        sepia_matrix = np.matrix([
            [0.131, 0.534, 0.272],
            [0.168, 0.686, 0.349],
            [0.189, 0.769, 0.393]
        ])
        res = cv2.transform(img_float, sepia_matrix)
        return np.clip(res, 0, 255).astype(np.uint8)

    def _add_heavy_physical_damage(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        result = img.copy()

        num_scratches = random.randint(15, 30)
        for _ in range(num_scratches):
            x1, y1 = random.randint(0, w), random.randint(0, h)
            x2, y2 = random.randint(0, w), random.randint(0, h)
            
            if random.random() < 0.7:
                color = (random.randint(220, 255), random.randint(220, 255), random.randint(220, 255))
            else:
                color = (random.randint(20, 50), random.randint(20, 50), random.randint(20, 50))
            
            thickness = random.randint(1, 3)
            cv2.line(result, (x1, y1), (x2, y2), color, thickness)

        noise = np.random.normal(0, 100, (h, w))
        smooth_noise = gaussian_filter(noise, sigma=30) 
        smooth_noise = np.stack([smooth_noise] * 3, axis=-1) 
        result = result.astype(np.float64) + (smooth_noise * 0.5)
        
        return np.clip(result, 0, 255).astype(np.uint8)

    def _add_extreme_blur(self, img: np.ndarray) -> np.ndarray:
        result = img.copy()
        h, w = result.shape[:2]

        scale = random.uniform(0.15, 0.3)
        small_img = cv2.resize(result, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)
        result = cv2.resize(small_img, (w, h), interpolation=cv2.INTER_CUBIC)

        k_gaussian = random.choice([15, 21, 25, 31]) 
        sigma = random.uniform(5, 10)
        result = cv2.GaussianBlur(result, (k_gaussian, k_gaussian), sigmaX=sigma)

        if random.random() < 0.5:
            motion_size = random.randint(20, 40)
            angle = random.randint(-90, 90)
            
            M = cv2.getRotationMatrix2D((motion_size / 2, motion_size / 2), angle, 1)
            kernel_motion = np.diag(np.ones(motion_size))
            kernel_motion = cv2.warpAffine(kernel_motion, M, (motion_size, motion_size))
            kernel_motion = kernel_motion / kernel_motion.sum()
            
            result = cv2.filter2D(result, -1, kernel_motion)

        return result


if __name__ == "__main__":
    transform = NewToOld(extreme_blur_prob=1.0, physical_damage_prob=1.0)
    
    test_img = np.zeros((256, 256, 3), dtype=np.uint8)
    for i in range(256):
        test_img[:, i] = (i, i, i)
        
    old = transform(test_img)
    cv2.imshow('Ultimate Extreme Degradation', old)
    print("按下任意鍵關閉視窗...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()