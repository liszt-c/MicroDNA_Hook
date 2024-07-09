###用于将所有序列文件合并起来
'''
合并的格式为：
>名称
ATCG

思路：
1.glob遍历cnvkit_do\\fa下的所有.txt文件
2.在cnvkit_do\connect打开一个记事本
3.循环读取glob文件，先写'>名称'
4.写入文件内容
'''
# 2024年7月9日 修正生成的fa序列格式
# -*- coding: UTF-8 -*-
import os
import glob


count = 0
ecc_path = glob.glob(r'.//cnvkit_do//fa//*_*_*.txt')
num_all  = len(ecc_path)
if os.path.exists('.//cnvkit_do//connect') == False:
    cmd2 = os.system('mkdir .//cnvkit_do//connect')
file = open(".//cnvkit_do//connect//all_ecc.fa",'w')
for i in ecc_path:
    count = count + 1
    write_line1 = '>num'+str(count)
    file.writelines(write_line1+'\n')
    with open(i, 'r') as f:
        read_DNA = f.read().replace('\n', '').replace('\r', '')
        file.write(read_DNA + '\n')
    print(count,'/',num_all)
file.close()
print(count)
