#读取短序列的概率，计算正例数量和正确率

Limit = 0.9
file_path = './result_out.txt'

def calculate_statistics(file_path, limit=0.9):
    total = 0
    positives = 0

    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) != 3:
                print(f"Warning: Skipping line '{line}' due to incorrect format.")
                continue
            
            probability = float(parts[2])
            total += 1
            
            if probability > limit:
                positives += 1

    accuracy = positives / total if total > 0 else 0
    return positives, total, accuracy

positives, total, accuracy = calculate_statistics(file_path, Limit)
print(f"Number of positives: {positives}")
print(f"Total number of data points: {total}")
print(f"Recognition accuracy: {accuracy * 100:.2f}%")