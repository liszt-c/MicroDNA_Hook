# 用于处理augustus得到的gff文件
# 清除提示信息和未预测到基因的序列信息
def process_gff_file(input_filename, output_filename):
    with open(input_filename, 'r') as infile, open(output_filename, 'w') as outfile:
        in_gene_section = False
        has_gene_prediction = False
        current_section = []
        
        for line in infile:
            if line.startswith("# ----- prediction on sequence number"):
                # 开始一个预测序列
                # 先手清除写入列表缓存
                current_section = []
                # 把序列指标设置为真
                # 基因指标设置为假
                in_gene_section = True
                has_gene_prediction = False
                # 不管后续有没有先把这一行写入缓存区列表中
                current_section.append(line)
            elif line.strip() == "###":
                # 如果检测到###序列
                # 说明是最后一段了
                if in_gene_section:
                    # 如果在序列内，并且检测到了基因，则把这一段写入
                    # 之后把缓存写入文件
                    if has_gene_prediction:
                        current_section.append(line)
                        outfile.writelines(current_section)
                        outfile.writelines('#\n') # 写一个#进去做分隔
                    # 开启下一个轮回，把基因检测标志重新打回false
                    in_gene_section = False    
            elif line.startswith('num'):
                # 检测到基因，则把基因标志设置为真，后续可以写入文件
                has_gene_prediction = True
                current_section.append(line)
            else:
                # 不是特殊行的话，写入缓存即可
                current_section.append(line)  

# Example usage:
input_file = 'ecc.gff'
output_file = 'ecc_clear.gff'
process_gff_file(input_file, output_file)