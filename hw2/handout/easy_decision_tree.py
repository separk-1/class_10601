import argparse
import numpy as np

class Node:
    '''
    노드 클래스는 결정 트리의 기본을 형성합니다.
    노드는 두 가지 주요 속성을 가질 수 있습니다:
        - attr: 노드에서 분할할 특성(attribute)
        - vote: 리프 노드에서의 예측값
    '''
    def __init__(self, attr=None, vote=None, left=None, right=None, left_labels=None, right_labels=None):
        self.attr = attr
        self.vote = vote
        self.left = left
        self.right = right
        self.left_labels = left_labels
        self.right_labels = right_labels
        
def calculate_entropy(labels):
    '''
    레이블의 엔트로피를 계산합니다.
    엔트로피는 정보의 불확실성을 측정합니다.
    '''
    label_counts = np.bincount(labels)  # 각 레이블의 빈도 계산
    total = len(labels)  # 레이블의 총 수
    
    # 엔트로피 계산
    entropy = 0
    for count in label_counts:
        if count > 0:
            probability = count / total
            entropy -= probability * np.log2(probability)
    return entropy

def calculate_mutual_information(attr, labels):
    '''
    주어진 특성(attr)과 레이블(labels)의 상호 정보를 계산합니다.
    상호 정보는 특성이 레이블에 대해 얼마나 정보를 제공하는지를 측정합니다.
    '''
    h_y = calculate_entropy(labels)  # 레이블의 엔트로피
    
    total = len(labels)  # 레이블의 총 수
    
    # 특성 값이 0인 인덱스
    attr_0_indices = (attr == 0)
    # 특성 값이 1인 인덱스
    attr_1_indices = (attr == 1)
    
    # 특성 값이 0일 때의 레이블
    attr_0_labels = labels[attr_0_indices]
    # 특성 값이 1일 때의 레이블
    attr_1_labels = labels[attr_1_indices]
    
    p_0 = len(attr_0_labels) / total  # 특성 값이 0일 확률
    p_1 = len(attr_1_labels) / total  # 특성 값이 1일 확률
    
    # 특성 값이 0일 때의 엔트로피
    h_x_0 = calculate_entropy(attr_0_labels) if len(attr_0_labels) > 0 else 0
    # 특성 값이 1일 때의 엔트로피
    h_x_1 = calculate_entropy(attr_1_labels) if len(attr_1_labels) > 0 else 0
    
    # 조건부 엔트로피 계산
    h_y_a = p_0 * h_x_0 + p_1 * h_x_1
    
    # 상호 정보 계산
    mutual_info = h_y - h_y_a
    
    return mutual_info

def calculate_error_rate(predictions, true_labels):
    '''
    예측값과 실제 레이블을 비교하여 오류율을 계산합니다.
    '''
    incorrect = np.sum(predictions != true_labels)  # 잘못된 예측의 수
    total = len(true_labels)  # 전체 레이블의 수
    
    error_rate = incorrect / total  # 오류율 계산
    return error_rate

def predict(tree, sample):
    '''
    주어진 샘플에 대해 결정 트리를 사용하여 예측을 수행합니다.
    '''
    if tree.vote is not None:
        return int(tree.vote[-1])  # 리프 노드에서 예측값 반환
    if sample[attr_list.index(tree.attr)] == 0:
        return predict(tree.left, sample)  # 왼쪽 자식 노드로 이동
    else:
        return predict(tree.right, sample)  # 오른쪽 자식 노드로 이동

def build_tree(attrs, labels, attr_list, depth=0, max_depth=None):
    if max_depth is not None and depth >= max_depth:
        majority_vote = 'Y=1' if sum(labels) >= len(labels) // 2 else 'Y=0'
        return Node(vote=majority_vote)

    # 상호 정보 계산
    mutual_infos = [calculate_mutual_information(attrs[:, i], labels) for i in range(attrs.shape[1])]
    best_attr_index = np.argmax(mutual_infos)
    best_attr = attr_list[best_attr_index]

    # 상호 정보가 0 이하인 경우, 다수결 투표로 리프 노드 생성
    if mutual_infos[best_attr_index] <= 0:
        majority_vote = 'Y=1' if sum(labels) >= len(labels) // 2 else 'Y=0'
        return Node(vote=majority_vote)

    # 특성 값에 따라 인덱스 분할
    left_indices = [i for i in range(len(labels)) if attrs[i, best_attr_index] == 0]
    right_indices = [i for i in range(len(labels)) if attrs[i, best_attr_index] == 1]

    # 인덱스가 비어있는 경우, 다수결 투표로 리프 노드 생성
    if len(left_indices) == 0 or len(right_indices) == 0:
        majority_vote = 'Y=1' if sum(labels) >= len(labels) // 2 else 'Y=0'
        return Node(vote=majority_vote)

    # 왼쪽과 오른쪽 자식 노드를 위한 데이터 분할
    left_attrs = attrs[left_indices, :]
    right_attrs = attrs[right_indices, :]
    left_labels = labels[left_indices]
    right_labels = labels[right_indices]

    # 현재 노드 생성 및 자식 노드 재귀적으로 구축
    root = Node(attr=best_attr, left_labels=left_labels, right_labels=right_labels)
    root.left = build_tree(left_attrs, left_labels, attr_list, depth + 1, max_depth)
    root.right = build_tree(right_attrs, right_labels, attr_list, depth + 1, max_depth)

    print("left_attrs", left_labels)
    print("right_attrs", right_labels)
    print(depth)
    return root

def count_labels(labels):
    '''
    레이블의 수를 세어 음성(neg) 및 긍정(pos) 레이블의 수를 반환합니다.
    '''
    print("labels", labels)
    pos_count = sum(labels)  # 긍정 레이블의 수
    neg_count = len(labels) - pos_count  # 음성 레이블의 수
    return neg_count, pos_count

def print_tree(node, file, depth=0):
    if node is None:
        return

    # 현재 노드의 들여쓰기 수준 설정
    indent = '| ' * (depth+1)
    
    # depth가 0일 때 전체 레이블 개수 출력
    if depth == 0:
        total_zeros = np.sum(node.left_labels == 0) + np.sum(node.right_labels == 0)
        total_ones = np.sum(node.left_labels == 1) + np.sum(node.right_labels == 1)
        file.write(f"[{total_zeros} 0/{total_ones} 1]\n")

    # 레이블 통계 계산
    if node.vote is not None:
        # 리프 노드일 경우
        return
    else:
        # 내부 노드일 경우
        left_labels = node.left_labels
        right_labels = node.right_labels

        # 레이블 통계 정보 생성
        num_zeros_left = np.sum(left_labels == 0)
        num_ones_left = np.sum(left_labels == 1)
        left_stats = f"[{num_zeros_left} 0/{num_ones_left} 1]"

        num_zeros_right = np.sum(right_labels == 0)
        num_ones_right = np.sum(right_labels == 1)
        right_stats = f"[{num_zeros_right} 0/{num_ones_right} 1]"

        # 노드 정보 출력
        file.write(f"{indent}{node.attr} = 0: {left_stats}\n")
        print_tree(node.left, file, depth + 1)
        file.write(f"{indent}{node.attr} = 1: {right_stats}\n")
        print_tree(node.right, file, depth + 1)
        
if __name__ == '__main__':
    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("train_input", type=str, help='path to training input .tsv file')
    parser.add_argument("test_input", type=str, help='path to the test input .tsv file')
    parser.add_argument("max_depth", type=int, help='maximum depth to which the tree should be built')
    parser.add_argument("train_out", type=str, help='path to output .txt file to which the feature extractions on the training data should be written')
    parser.add_argument("test_out", type=str, help='path to output .txt file to which the feature extractions on the test data should be written')
    parser.add_argument("metrics_out", type=str, help='path of the output .txt file to which metrics such as train and test error should be written')
    parser.add_argument("print_out", type=str, help='path of the output .txt file to which the printed tree should be written')
    args = parser.parse_args()

    # Read train data    
    train_labels = []
    train_attrs = []
    with open(args.train_input, 'r') as file:
        first_line = file.readline().strip()
        attr_list = first_line.split()[:-1]
        for line in file:
            values = line.strip().split('\t')
            train_labels.append(int(values[-1]))
            train_attrs.append([int(val) for val in values[:-1]])
            
    train_labels = np.array(train_labels)
    train_attrs = np.array(train_attrs)
    
    #print(train_labels)
    #print(train_attrs)
    #print(attr_list)
    
    #print(train_attrs.shape)
    # Build tree
    tree_root = build_tree(train_attrs, train_labels, attr_list, 0, args.max_depth)

    # Print tree to file
    with open(args.print_out, 'w') as file:
        print_tree(tree_root, file)

    # Predictions for train and test
    train_predictions = [predict(tree_root, sample) for sample in train_attrs]
    test_labels = []
    test_attrs = []
    with open(args.test_input, 'r') as file:
        next(file)  # skip header
        for line in file:
            values = line.strip().split('\t')
            test_labels.append(int(values[-1]))
            test_attrs.append([int(val) for val in values[:-1]])
            
    test_predictions = [predict(tree_root, sample) for sample in test_attrs]

    # Calculate error rates
    train_error = calculate_error_rate(train_predictions, train_labels)
    test_error = calculate_error_rate(test_predictions, test_labels)

    # Write predictions to output files
    with open(args.train_out, 'w') as train_file:
        for pred in train_predictions:
            train_file.write(f"{pred}\n")

    with open(args.test_out, 'w') as test_file:
        for pred in test_predictions:
            test_file.write(f"{pred}\n")

    # Write metrics
    with open(args.metrics_out, 'w') as metrics_file:
        metrics_file.write(f"error(train): {train_error:.6f}\n")
        metrics_file.write(f"error(test): {test_error:.6f}\n")