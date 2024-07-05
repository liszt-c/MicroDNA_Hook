# -*- coding: UTF-8 -*-
import os
import sys
from glob import glob
import argparse
import platform

if platform.system().lower() == 'windows':
    print("windows")
    python_ = 'python'
elif platform.system().lower() == 'linux':
    print("linux")
    python_ = 'python3'


def run_cnvkit():
    search = glob(r'.//cnvkit_do//out//result.call.cns')
    if search == []:
        cmd0 = os.system(str(python_) + ' ./cnvkit_do/cnvkit_tool.py')

    #index
    search2 = glob(r'./cnvkit_do//*.bt2')
    if search2 == []:
        cmd0 = os.system('bowtie2-build -f .//cnvkit_do//hg19.fa .//cnvkit_do//hg19')
        print(cmd0)

    file  =  open(".//cnvkit_do//out//result.call.cns",  "r")
    cout = 0
    if os.path.exists('.//cnvkit_do//fa') == False:
        cmd2 = os.system('mkdir .//cnvkit_do//fa')
    while True:
        line  =  file.readline()
        if  not  line:
            break
        line2 = line.split()
        if line2[0][:3] == 'chr':
            cmd1 = os.system('samtools faidx .//cnvkit_do//hg19.fa '+str(line2[0])+':'+str(line2[1])+'-'+str(line2[2])+' > .//cnvkit_do//fa/'+'cnvkit_'+str(cout)+'.fa')
        cout = cout+1    
    file.close()
    return 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'RUN')
    parser.add_argument('--model',default = "6.pth", type=str,
                        help='model name, The model needs to be in the folder ./save')
    parser.add_argument('--run',default = "run_v11.3.py", type=str,
                        help='run flie name. default run_v11.3.py')
    parser.add_argument('--limit',default = "0.95",type=str,
                        help='parameter:strict，normal，relax. relax pattern can identify more but with less precision but strict pattern opposite.')
    args = parser.parse_args()
    model_name = args.model

    search0 = glob(r'.//cnvkit_do//fa//*fa')
    if search0 ==[]:
        print('cnvkit runing')
        ###run cnvkit_tool.py if not result file
        x = run_cnvkit()
    else:
        print('*.fa in .//cnvkit_do//fa exit!')
    file_name = args.run
    limite = args.limit
    ###run long_segment pr
    cmd3 = os.system(str(python_)+' '+str(file_name)+' --pattern long_segment --model '+str(model_name)+' --file_path .//cnvkit_do//fa --limit '+str(limite))









