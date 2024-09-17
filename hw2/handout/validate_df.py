import pandas as pd

# 파일 경로에 맞게 변경
file_path = 'small_train.tsv'

# TSV 파일을 DataFrame으로 읽어오기
df = pd.read_csv(file_path, sep='\t')

# 전체 데이터에서 heart_disease 레이블 분포 확인
total_neg = df[df['heart_disease'] == 0].shape[0]
total_pos = df[df['heart_disease'] == 1].shape[0]
print(f"Total: [{total_neg} 0/{total_pos} 1]")

# 1단계: chest_pain = 0인 데이터 분리
df_left = df[df['chest_pain'] == 0]
print(df_left)

print(df_left[df_left['thalassemia']==0])
#df_right = df[df['chest_pain'] == 1]
'''

# chest_pain = 0에서 heart_disease 분포 확인
left_neg = df_left[df_left['heart_disease'] == 0].shape[0]
left_pos = df_left[df_left['heart_disease'] == 1].shape[0]
print(f"| chest_pain = 0: [{left_neg} 0/{left_pos} 1]")

# chest_pain = 1에서 heart_disease 분포 확인
right_neg = df_right[df_right['heart_disease'] == 0].shape[0]
right_pos = df_right[df_right['heart_disease'] == 1].shape[0]
print(f"| chest_pain = 1: [{right_neg} 0/{right_pos} 1]")

# 2단계: thalassemia = 0인 데이터 분리 (chest_pain = 0)
df_left_left = df_left[df_left['thalassemia'] == 0]
df_left_right = df_left[df_left['thalassemia'] == 1]

# chest_pain = 0, thalassemia = 0에서 heart_disease 분포 확인
left_left_neg = df_left_left[df_left_left['heart_disease'] == 0].shape[0]
left_left_pos = df_left_left[df_left_left['heart_disease'] == 1].shape[0]
print(f"|| thalassemia = 0: [{left_left_neg} 0/{left_left_pos} 1]")

# chest_pain = 0, thalassemia = 1에서 heart_disease 분포 확인
left_right_neg = df_left_right[df_left_right['heart_disease'] == 0].shape[0]
left_right_pos = df_left_right[df_left_right['heart_disease'] == 1].shape[0]
print(f"|| thalassemia = 1: [{left_right_neg} 0/{left_right_pos} 1]")

# 2단계: thalassemia = 0인 데이터 분리 (chest_pain = 1)
df_right_left = df_right[df_right['thalassemia'] == 0]
df_right_right = df_right[df_right['thalassemia'] == 1]

# chest_pain = 1, thalassemia = 0에서 heart_disease 분포 확인
right_left_neg = df_right_left[df_right_left['heart_disease'] == 0].shape[0]
right_left_pos = df_right_left[df_right_left['heart_disease'] == 1].shape[0]
print(f"|| thalassemia = 0: [{right_left_neg} 0/{right_left_pos} 1]")

# chest_pain = 1, thalassemia = 1에서 heart_disease 분포 확인
right_right_neg = df_right_right[df_right_right['heart_disease'] == 0].shape[0]
right_right_pos = df_right_right[df_right_right['heart_disease'] == 1].shape[0]
print(f"|| thalassemia = 1: [{right_right_neg} 0/{right_right_pos} 1]")'''