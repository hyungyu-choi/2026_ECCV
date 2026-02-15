import os
import glob
import numpy as np

def analyze_cholec80_frames(label_dir, video_ids):
    """
    라벨 파일을 읽어서 프레임 통계를 계산합니다.
    """
    frame_counts = []
    
    for v_id in video_ids:
        # 파일 패턴: video01-phase.txt, video41-phase.txt 등
        file_path = os.path.join(label_dir, f"video{v_id:02d}-phase.txt")
        
        if not os.path.exists(file_path):
            # 만약 파일명이 'video01.txt' 형태라면 아래 주석을 해제하세요.
            # file_path = os.path.join(label_dir, f"video{v_id:02d}.txt")
            if not os.path.exists(file_path):
                print(f"Warning: 파일을 찾을 수 없음: {file_path}")
                continue
        
        count = 0
        with open(file_path, 'r') as f:
            # 헤더 제외 (Frame Phase)
            lines = f.readlines()[1:]
            for line in lines:
                if line.strip():  # 빈 줄이 아니면 카운트
                    count += 1
        
        frame_counts.append(count)
        # print(f"Video {v_id:02d}: {count} frames") # 개별 비디오 확인용

    if not frame_counts:
        return 0, 0, 0

    total_frames = sum(frame_counts)
    avg_frames = total_frames / len(frame_counts)
    num_videos = len(frame_counts)
    
    return total_frames, avg_frames, num_videos

# 설정 (본인의 경로에 맞게 수정하세요)
train_label_dir = '../../code/Dataset/cholec80/phase_annotations/training_set_1fps'
test_label_dir = '../../code/Dataset/cholec80/phase_annotations/test_set_1fps'

# 1. Training Set (1~40번)
train_ids = list(range(1, 41))
t_total, t_avg, t_num = analyze_cholec80_frames(train_label_dir, train_ids)

# 2. Test Set (41~80번)
test_ids = list(range(41, 81))
s_total, s_avg, s_num = analyze_cholec80_frames(test_label_dir, test_ids)

print("="*50)
print(f"훈련 데이터 (Videos 01-40) 통계:")
print(f" - 대상 비디오 수: {t_num}개")
print(f" - 전체 프레임 수: {t_total:,}개")
print(f" - 비디오당 평균 프레임: {t_avg:.2f}개")
print("="*50)
print(f"테스트 데이터 (Videos 41-80) 통계:")
print(f" - 대상 비디오 수: {s_num}개")
print(f" - 전체 프레임 수: {s_total:,}개")
print(f" - 비디오당 평균 프레임: {s_avg:.2f}개")
print("="*50)
print(f"최종 총합 (Total): {t_total + s_total:,} 프레임")