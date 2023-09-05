import os
import sys
import openpyxl

###参数定义
SeqName = 'GRCh38.p13.genome'	#序列名称
AllChar = 25987		#ecDNA片段总个数
First = 420729		#在表格中的起始位置
End = 446715		#在表格中的终止位置
Class = 'fa'		#文件类型

###samtools建立索引
os.system('samtools faidx '+SeqName+'.fa')

###格式化bash
os.system('dos2unix SamtoolBash.sh')

###函数调用bash
def bash(SeqName,CharNum,Left,Right,CharName,Class):
	state=os.system('./SamtoolBash.sh '+SeqName+' '+CharNum+' '+str(Left)+' '+str(Right)+' '+str(CharName)+' '+Class)
	return state

###读取excel
workbook = openpyxl.load_workbook('eccDNA_Homo sapiens.xlsx')
sheet = workbook.active  	# 获取活动表

###循环主体，按行获取表格信息
n,k = 0,0			#计数器,samtools返回标志位
x,y,z=0,0,0
h,w,q=0,0,0
for i in sheet.iter_rows(min_row=First, max_row=End, min_col=6,max_col=8):

    for j in i:
        h,w,q,x,y,z=w,q,x,y,z,j.value       ###参数传递两轮后
        print(type(h))
        print(h)
        ###判断两轮
        if(type(h) is str):
            ###左边区间是q，右边区间是y
            if(h==x and y>q ):
                if(y-q>600):               ###中间距离大于600，切最中间的400
                    #lef,rig=q,y
                    cycle = y-q
                    half = int(cycle/2)
                    lef = q + half -200
                    rig = lef + 400
                    n = n + 1
                    print(SeqName,x,lef,rig,n)
                    k=bash(SeqName,x,lef,rig,n,Class)
                    print(k)
'''
                if(y-q<=1000):
                    n = n+1
                    print(SeqName,x,q,y,n)
                    k=bash(SeqName,x,q,y,n,Class)
                    print(k)
'''