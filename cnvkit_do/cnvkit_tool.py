import os
import sys
from glob import glob
import argparse


SeqName = 'SRR61569A'
thread = '-p 10'
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'RUN')
    parser.add_argument('--thread',default = "-p 12", type=str,
                        help='thread numbers.')
    parser.add_argument('--begin_with',default = "SRA", type=str,
                        help='SRA ')
    parser.add_argument('--limit',default = "0.95",type=str,
                        help='parameter:strict，normal，relax. relax pattern can identify more but with less precision but strict pattern opposite.')
    args = parser.parse_args()


    search = 0
    search = glob(r'.//*.sra')
    if search != []:
        for i in search :
            cmd1 = os.system('fasterq-dump --split-3 '+str(search))
            print(cmd1)
            SeqName = str(search)
            
    search2 = 0
    search2 = glob(r'.//*.bt2')
    if search2 == []:
        cmd0 = os.system('bowtie2-build -f hg19.fa hg19')
        print(cmd0)
    # 对于单端测序
    # bowtie2 -p 10 -x hg19 -U input.fq | samtools sort -O bam -@ 12 -o - > output.bam
    cmd2 = os.system('bowtie2 -p 12 -x hg19 -1 '+str(SeqName)+'_1.fastq -2 '+str(SeqName)+'_2.fastq | samtools sort -@12 -o '+str(SeqName)+'.bam')
    print(cmd2)

    ##-p '+str(thread)
    cmd3 = os.system('cnvkit.py batch -m wgs -r hg19_cnvkit_filtered_ref.cnn '+str(thread)+' -d ./out '+str(SeqName)+'.bam')
    print(cmd3)

    cmd4 = os.system('cnvkit.py segment ./out/'+str(SeqName)+'.cnr '+str(thread)+' -m cbs -o ./out/result.cns')
    print(cmd4)

    cmd5 = os.system('cnvkit.py call ./out/result.cns -o ./out/result.call.cns')
    print(cmd5)

