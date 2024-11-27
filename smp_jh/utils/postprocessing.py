import torch
import numpy as np
from scipy.ndimage import binary_dilation
from config.config import Config

class AnatomicalPostProcessor:
    def __init__(self, threshold=0.5):
        self.threshold = threshold
        
        # Config에서 클래스 인덱스 가져오기
        self.class2idx = Config.CLASS2IND
        
        # 손가락 그룹 (클래스 이름 기반)
        self.finger_groups = [
            ['finger-1', 'finger-2', 'finger-3', 'finger-4'],  # 첫번째 손가락
            ['finger-5', 'finger-6', 'finger-7', 'finger-8'],  # 두번째 손가락
            ['finger-9', 'finger-10', 'finger-11', 'finger-12'],  # 세번째 손가락
            ['finger-13', 'finger-14', 'finger-15', 'finger-16'],  # 네번째 손가락
            ['finger-17', 'finger-18', 'finger-19']  # 다섯번째 손가락
        ]
        
        # 클래스 이름을 인덱스로 변환
        self.finger_group_indices = [
            [self.class2idx[name] for name in group]
            for group in self.finger_groups
        ]
        
        # 겹칠 수 있는 손목 뼈 쌍
        self.overlapping_pairs = [
            ('Trapezium', 'Trapezoid'),
            ('Triquetrum', 'Pisiform'),
            ('Scaphoid', 'Lunate'),
            ('Lunate', 'Triquetrum'),
            ('Capitate', 'Hamate'),
            ('Lunate', 'Capitate'),  # 추가
            ('Scaphoid', 'Capitate')  # 추가
        ]
        
        self.overlapping_indices = [
            (self.class2idx[bone1], self.class2idx[bone2])
            for bone1, bone2 in self.overlapping_pairs
        ]
        
    def __call__(self, prediction):
        """
        Args:
            prediction: (B, C, H, W) 형태의 모델 예측값
        Returns:
            processed_pred: 후처리된 예측값
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
        for finger_indices in self.finger_group_indices:
            self._process_finger_group(processed, finger_indices)
        
        # 2. 손목 뼈 겹침 처리
        for bone1_idx, bone2_idx in self.overlapping_indices:
            self._process_overlapping_bones(processed, original, bone1_idx, bone2_idx)
        
        return processed
    
    def _process_finger_group(self, processed, finger_indices):
        """손가락 마디 연속성 처리"""
        for i in range(len(finger_indices)-1):
            curr_conf = processed[finger_indices[i]]
            next_conf = processed[finger_indices[i+1]]
            
            # 높은 신뢰도 영역 확인
            curr_high_conf = curr_conf > 0.7  # 높은 신뢰도 임계값
            next_high_conf = next_conf > 0.7
            
            if curr_high_conf.any() and next_high_conf.any():
                if not self._check_connectivity(curr_high_conf, next_high_conf):
                    # 연결이 필요한 경우 신뢰도 값을 보간
                    curr_new, next_new = self._connect_segments_with_confidence(
                        curr_conf, next_conf, curr_high_conf, next_high_conf)
                    processed[finger_indices[i]] = curr_new
                    processed[finger_indices[i+1]] = next_new

    def _connect_segments_with_confidence(self, conf1, conf2, mask1, mask2):
        """신뢰도 값을 보존하면서 분절된 마스크 연결"""
        # numpy로 변환
        conf1_np = conf1.cpu().numpy()
        conf2_np = conf2.cpu().numpy()
        mask1_np = mask1.cpu().numpy()
        mask2_np = mask2.cpu().numpy()
        
        # 팽창
        dilated1 = binary_dilation(mask1_np, iterations=1)
        dilated2 = binary_dilation(mask2_np, iterations=1)
        
        # 연결 영역 찾기
        connection = dilated1 & dilated2
        
        # 연결 영역의 신뢰도 값을 보간
        if connection.any():
            # 두 마스크의 평균 신뢰도로 연결 영역 채우기
            mean_conf = (conf1_np.mean() + conf2_np.mean()) / 2
            conf1_np[connection] = max(conf1_np[connection].mean(), mean_conf * 0.8)
            conf2_np[connection] = max(conf2_np[connection].mean(), mean_conf * 0.8)
        
        # torch tensor로 변환하여 반환
        return (torch.from_numpy(conf1_np).to(conf1.device), 
                torch.from_numpy(conf2_np).to(conf2.device))
    
    def _process_overlapping_bones(self, processed, original, bone1_idx, bone2_idx):
        """겹치는 손목 뼈 처리 - 신뢰도 기반"""
        conf1 = processed[bone1_idx]
        conf2 = processed[bone2_idx]
        
        # 높은 신뢰도 영역 확인
        high_conf1 = conf1 > 0.7
        high_conf2 = conf2 > 0.7
        
        overlap = high_conf1 & high_conf2
        if overlap.any():
            # 겹치는 영역의 신뢰도 조정
            mean_conf1 = conf1[overlap].mean()
            mean_conf2 = conf2[overlap].mean()
            
            if mean_conf1 > 0.3 and mean_conf2 > 0.3:
                # 둘 다 높은 신뢰도: 약간 감소된 신뢰도로 유지
                processed[bone1_idx][overlap] *= 0.9
                processed[bone2_idx][overlap] *= 0.9
            else:
                # 하나만 높은 경우: 더 높은 신뢰도를 가진 쪽 유지
                if mean_conf1 > mean_conf2:
                    processed[bone2_idx][overlap] *= 0.1
                else:
                    processed[bone1_idx][overlap] *= 0.1
    
    @staticmethod
    def _check_connectivity(mask1, mask2):
        """두 마스크의 연결성 확인"""
        dilated = binary_dilation(mask1.cpu().numpy(), iterations=2)
        return np.any(dilated & mask2.cpu().numpy())
    
    @staticmethod
    def _connect_segments(mask1, mask2):
        """분절된 마스크 연결"""
        # 양방향 팽창으로 자연스러운 연결
        dilated1 = binary_dilation(mask1.cpu().numpy(), iterations=1)
        dilated2 = binary_dilation(mask2.cpu().numpy(), iterations=1)
        
        connection = dilated1 & dilated2
        new_mask1 = torch.from_numpy(mask1.cpu().numpy() | connection).to(mask1.device)
        new_mask2 = torch.from_numpy(mask2.cpu().numpy() | connection).to(mask2.device)
        
        return new_mask1, new_mask2
    
    def visualize_postprocessing(self, processed, original, batch_idx=0, save_dir='postprocessing_vis'):
        """후처리 전후 결과 시각화"""
        import matplotlib.pyplot as plt
        import os
        
        os.makedirs(save_dir, exist_ok=True)
        
        # 손가락 연결성 시각화
        for finger_idx, finger_group in enumerate(self.finger_group_indices):
            plt.figure(figsize=(15, 5))
            
            # 원본 신뢰도
            plt.subplot(131)
            orig_finger = torch.zeros_like(original[finger_group[0]])
            for idx in finger_group:
                orig_finger = torch.maximum(orig_finger, original[idx])
            plt.imshow(orig_finger.cpu().numpy(), cmap='jet', vmin=0, vmax=1)
            plt.colorbar()
            plt.title(f'Original Confidence - Finger {finger_idx+1}')
            
            # 후처리 신뢰도
            plt.subplot(132)
            proc_finger = torch.zeros_like(processed[finger_group[0]])
            for idx in finger_group:
                proc_finger = torch.maximum(proc_finger, processed[idx])
            plt.imshow(proc_finger.cpu().numpy(), cmap='jet', vmin=0, vmax=1)
            plt.colorbar()
            plt.title(f'Processed Confidence - Finger {finger_idx+1}')
            
            # 차이
            plt.subplot(133)
            diff = proc_finger.cpu().numpy() - orig_finger.cpu().numpy()
            plt.imshow(diff, cmap='bwr', vmin=-1, vmax=1)
            plt.colorbar()
            plt.title('Confidence Difference')
            
            plt.savefig(os.path.join(save_dir, f'finger_{finger_idx+1}_batch_{batch_idx}.png'))
            plt.close()
        
        # 손목 뼈 겹침 시각화
        for (bone1_name, bone2_name), (bone1_idx, bone2_idx) in zip(self.overlapping_pairs, self.overlapping_indices):
            plt.figure(figsize=(20, 5))
            
            # 원본 신뢰도 (bone1)
            plt.subplot(141)
            plt.imshow(original[bone1_idx].cpu().numpy(), cmap='jet', vmin=0, vmax=1)
            plt.colorbar()
            plt.title(f'Original {bone1_name} Confidence')
            
            # 원본 신뢰도 (bone2)
            plt.subplot(142)
            plt.imshow(original[bone2_idx].cpu().numpy(), cmap='jet', vmin=0, vmax=1)
            plt.colorbar()
            plt.title(f'Original {bone2_name} Confidence')
            
            # 후처리 신뢰도 (bone1)
            plt.subplot(143)
            plt.imshow(processed[bone1_idx].cpu().numpy(), cmap='jet', vmin=0, vmax=1)
            plt.colorbar()
            plt.title(f'Processed {bone1_name} Confidence')
            
            # 후처리 신뢰도 (bone2)
            plt.subplot(144)
            plt.imshow(processed[bone2_idx].cpu().numpy(), cmap='jet', vmin=0, vmax=1)
            plt.colorbar()
            plt.title(f'Processed {bone2_name} Confidence')
            
            plt.savefig(os.path.join(save_dir, f'{bone1_name}_{bone2_name}_batch_{batch_idx}.png'))
            plt.close()
            
            # 겹침 영역 상세 시각화
            plt.figure(figsize=(15, 5))
            
            # 원본 겹침
            plt.subplot(131)
            high_conf1 = original[bone1_idx] > 0.7
            high_conf2 = original[bone2_idx] > 0.7
            overlap_orig = (high_conf1 & high_conf2).cpu().numpy()
            plt.imshow(overlap_orig, cmap='gray')
            plt.title('Original Overlap')
            
            # 후처리 겹침
            plt.subplot(132)
            high_conf1 = processed[bone1_idx] > 0.7
            high_conf2 = processed[bone2_idx] > 0.7
            overlap_proc = (high_conf1 & high_conf2).cpu().numpy()
            plt.imshow(overlap_proc, cmap='gray')
            plt.title('Processed Overlap')
            
            # 차이
            plt.subplot(133)
            diff = overlap_proc.astype(int) - overlap_orig.astype(int)
            plt.imshow(diff, cmap='bwr')
            plt.title('Overlap Difference')
            
            plt.savefig(os.path.join(save_dir, f'{bone1_name}_{bone2_name}_overlap_batch_{batch_idx}.png'))
            plt.close()