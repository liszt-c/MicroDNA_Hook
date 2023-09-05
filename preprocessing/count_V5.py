import os
import sys
import openpyxl

###参数定义
SeqName = 'hg19'	#序列名称
FileName = 'GSM1678006_Human-ES2-microDNA.xlsx'
AllChar = 25987		#ecDNA片段总个数
First_row = 2		#在表格中的起始位置
End_row = 114616		#在表格中的终止位置
First_col = 1
End_col = 3
Class = 'fa'		#文件类型

norm_length = 400	#归一化扩展长度

###samtools建立索引
#os.system('samtools faidx '+SeqName+'.fa')

###格式化bash
os.system('dos2unix SamtoolBash.sh')

###函数调用bash
def bash(SeqName,CharNum,Left,Right,CharName,Class):
	state=os.system('./SamtoolBash.sh '+SeqName+' '+CharNum+' '+str(Left)+' '+str(Right)+' '+str(CharName)+' '+Class)
	return state

###读取excel
workbook = openpyxl.load_workbook(FileName)
sheet = workbook.active  	# 获取活动表

###
num_under200 = 0
num_200_400 = 0
num_400_600 = 0
num_600_800 = 0
num_800_1K = 0
num_than_1K = 0
###循环主体，按行获取表格信息
n,k = 0,0			#计数器
for i in sheet.iter_rows(min_row=First_row, max_row=End_row, min_col=First_col,max_col=End_col):
	x,y,z=0,0,0
	for j in i:
		x,y,z=y,z,j.value
	if(type(x) is str):					###上下求索补足
		if(z-y < norm_length):
			lef,rig = y,z
			half = int((rig-lef)/2)		###int()向下取整
			fill = norm_length/2 - half
			lef = lef - fill
			rig = rig + fill
			if(rig-lef == norm_length+1):
				rig = rig -1
			n = n + 1
			print(SeqName,x,lef,rig,n,z-y)
			k = bash(SeqName,x,lef,rig,n,Class)
			length = z - y
			print(k)
		else:
			n = n + 1
			print(SeqName,x,y,z,n,z-y)
			k=bash(SeqName,x,y,z,n,Class)
			print(k)
			length = z - y


		if(length<200):
			num_under200 = num_under200 + 1
		if(200<=length<400):
			num_200_400 = num_200_400 + 1
		if(400<=length<600):
			num_400_600 = num_400_600 + 1
		if(600<=length<800):
			num_600_800 = num_600_800 + 1
		if(800<=length<1000):
			num_800_1K = num_800_1K + 1
		if(1000<=length):
			num_than_1K = num_than_1K +	1
print('num_under200',num_under200,num_under200/n)
print('num_200_400',num_200_400,num_200_400/n)
print('num_400_600',num_400_600,num_400_600/n)
print('num_600_800',num_600_800,num_600_800/n)
print('num_800_1000',num_800_1K,num_800_1K/n)
print('num_than_1K',num_than_1K,num_than_1K/n)






















