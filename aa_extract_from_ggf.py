# 从augustus得到的gff文件提取预测的转录组蛋白序列
def extract_protein_sequences(input_file_path, output_file_path):
    with open(input_file_path, 'r') as file:
        lines = file.readlines()
        sequences = []
        in_seq = False
        current_sequence = ''
        for line in lines:
            if "[" in line and "]" in line:
                # 如果同时检测到[]，则直接写入序列列表
                # 找到[]的位置，之后切片写入
                start = line.find('[') + 1
                end = line.find(']')
                current_sequence = line[start:end].strip()
                sequences.append(current_sequence)
            elif "[" in line and "]" not in line:
                # 如果检测到[，而没有]，则开始计算
                in_seq = True
                start = line.find('[') + 1
                current_line = line[start:].strip()
                current_sequence = current_line
            elif "]" in line and "[" not in line:
                end = line.find(']')
                # 注意清除前面的#
                current_sequence += line[2:end].strip()
                sequences.append(current_sequence)
                # 清空当前序列
                current_sequence = None
                # 重新把序列内标志打回否
                in_seq = False
            else:
                if line:
                    if in_seq:
                        line = line[2:]
                        current_sequence += line.strip()
                        # 如果在[后面，且本行没有]
                        # 如果line不为空，这两个其实用第二个就可以
                        # 删除前面的“# ”
                        # 存入current_sequence缓存中


    # Write the sequences to the output file
    with open(output_file_path, 'w') as out_file:
        for i, seq in enumerate(sequences):
            out_file.write(f"> seq{i+1}\n")
            out_file.write(seq + "\n")


input_file = "ecc.gff"
output_file = "ecc_clear.aa"
extract_protein_sequences(input_file, output_file)