import matplotlib.pyplot as plt

# 데이터 입력 (Max-Depth, Train Error, Test Error)
max_depth = [0, 1, 2, 3, 4, 5, 6, 7]
train_error = [0.4900, 0.2150, 0.2150, 0.1400, 0.1300, 0.0950, 0.1000, 0.1150]
test_error = [0.4021, 0.2784, 0.3299, 0.1753, 0.2474, 0.2474, 0.2577, 0.2680]

# 글꼴 설정
plt.rcParams["font.family"] = "Times New Roman"

# 그래프 그리기
plt.plot(max_depth, train_error, label="Train Error", marker='o', color='skyblue')
plt.plot(max_depth, test_error, label="Test Error", marker='o', color='gray')

# 데이터포인트마다 값 표시
for i, v in enumerate(train_error):
    plt.text(max_depth[i], v, f"{v:.4f}", ha='right', va='bottom', fontsize=9)
for i, v in enumerate(test_error):
    plt.text(max_depth[i], v, f"{v:.4f}", ha='left', va='bottom', fontsize=9)

# 라벨 및 제목 설정
plt.xlabel("Max Depth")
plt.ylabel("Error Rate")
plt.title("Train and Test Error vs Max Depth")

# 범례 추가
plt.legend()

# 격자 표시
plt.grid(True)

# 그래프 저장
plt.savefig('train_test_error_graph.png', dpi=300)

# 그래프 출력
plt.show()