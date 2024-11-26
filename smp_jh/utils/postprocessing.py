import torch
import numpy as np
from scipy.ndimage import binary_dilation

class AnatomicalPostProcessor:
    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.finger_groups = [
            [0, 1, 2, 3],      # 첫번째 손가락
            [4, 5, 6, 7],      # 두번째 손가락
            [8, 9, 10, 11],    # 세번째 손가락
            [12, 13, 14, 15],  # 네번째 손가락
            [16, 17, 18]       # 다섯번째 손가락
        ]
        self.overlapping_pairs = [
            (19, 20),  # Trapezium-Trapezoid
            (25, 26)   # Triquetrum-Pisiform
        ]
    
    def __call__(self, prediction):
        """
        prediction: (B, C, H, W) 형태의 모델 예측값
        C: 클래스 수 (29개: 19개 손가락 마디 + 8개 손목 뼈 + radius + ulna)
        """
        return self.process(prediction)
    
    def process(self, prediction):
        processed = prediction.clone()
        batch_size = prediction.shape[0]
        
        for b in range(batch_size):
            processed[b] = self._process_single_image(processed[b], prediction[b])
        
        return processed
    
    def _process_single_image(self, processed, original):
        # 1. 손가락 연속성 처리
        for finger in self.finger_groups:
            for i in range(len(finger)-1):
                curr_mask = processed[finger[i]] > self.threshold
                next_mask = processed[finger[i+1]] > self.threshold
                
                if not self._check_connectivity(curr_mask, next_mask):
                    processed[finger[i]], processed[finger[i+1]] = \
                        self._connect_segments(curr_mask, next_mask)
        
        # 2. 손목 뼈 겹침 처리
        for bone1, bone2 in self.overlapping_pairs:
            mask1 = processed[bone1] > self.threshold
            mask2 = processed[bone2] > self.threshold
            
            overlap = mask1 & mask2
            if overlap.any():
                processed[bone1][overlap] = \
                    1.0 if original[bone1][overlap].mean() > original[bone2][overlap].mean() else 0.0
                processed[bone2][overlap] = \
                    1.0 if original[bone2][overlap].mean() > original[bone1][overlap].mean() else 0.0
        
        # 3. Radius-Ulna 관계 처리
        radius_mask = processed[-2] > self.threshold
        ulna_mask = processed[-1] > self.threshold
        processed[-2], processed[-1] = self._adjust_radius_ulna(radius_mask, ulna_mask)
        
        return processed
    
    @staticmethod
    def _check_connectivity(mask1, mask2):
        """두 마스크가 서로 연결되어 있는지 확인"""
        dilated = binary_dilation(mask1.cpu().numpy(), iterations=2)
        return np.any(dilated & mask2.cpu().numpy())
    
    @staticmethod
    def _connect_segments(mask1, mask2):
        """두 분절을 연결"""
        dilated1 = binary_dilation(mask1.cpu().numpy(), iterations=1)
        dilated2 = binary_dilation(mask2.cpu().numpy(), iterations=1)
        
        connection = dilated1 & dilated2
        new_mask1 = mask1.cpu().numpy() | connection
        new_mask2 = mask2.cpu().numpy() | connection
        
        return torch.from_numpy(new_mask1), torch.from_numpy(new_mask2)
    
    @staticmethod
    def _adjust_radius_ulna(radius_mask, ulna_mask):
        """Radius와 Ulna의 위치 관계 조정"""
        # TODO: 구체적인 해부학적 규칙 구현
        return radius_mask, ulna_mask