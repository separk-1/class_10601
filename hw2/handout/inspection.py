import argparse
import numpy as np

def calculate_entropy(labels):
    label_counts = np.bincount(labels)
    total = len(labels)
    
    # calculate entropy
    entropy = 0
    for count in label_counts:
        if count > 0:
            probability = count / total
            entropy -= probability * np.log2(probability)
    return entropy

def calculate_error_rate(labels):
    majority_label = np.bincount(labels).argmax()
    
    # calculate error rate
    total = len(labels)
    incorrect = np.sum(labels != majority_label)
    error_rate = incorrect / total
    return error_rate

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str, help="path to the input .tsv file")
    parser.add_argument("output", type=str, help="path to the output .txt file")
    args = parser.parse_args()

    labels = []
    with open(args.input, 'r') as file:
        next(file)
        for line in file:
            labels.append(int(line.strip().split('\t')[-1]))
    
    labels = np.array(labels)
    
    entropy = calculate_entropy(labels)
    
    error_rate = calculate_error_rate(labels)

    with open(args.output, 'w') as out_file:
        out_file.write(f"entropy: {entropy:.6f}\n")
        out_file.write(f"error: {error_rate:.6f}\n")

if __name__ == '__main__':
    main()