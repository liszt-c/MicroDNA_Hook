# MicroDNA Hook

### 文件夹结构
``` 
-|datasets- |eccDNA	#放入eccDNA序列，可以">"注释开头
 |		|
 | 		|otherDNA	#放入其他DNA序列，可以">"注释开头
 |
-|cnvkit_do- |out	#cnvkit运行文件
 |		|
 | 		|fa	#运行结果文件，并将存储基于CNVkit的识别结果
 |
-|identify- |fatsa_to_identify	#短序列识别方法默认文件夹
 |		|
 | 		|long_segment_to_identify#长序列识别方法默认文件夹
 |
 |ROC			#ROC输出及图像绘制文件夹
 |
-|run			#训练中间参数储存文件夹
 |
-|save		#模型储存文件夹

文件
-|preprocessing-	|SamtoolBash.sh	#bash命令，提供samtools接口
 |			|
 |			|count*.py		#读取xcel文档并调用SamtoolBash.sh切割eccDNA序列
 |			|
 |			|cout_other*.py	#读取xcel文档并调用SamtoolBash.sh切割其他DNA序列
 |
-|dataprocess*.py	#读取DNA序列，将序列转为矩阵
 |
-|dataloader*.py	#读取转化后的矩阵，通过 pytorch构建datasets和dataloader，并分出20%作为测试集
 |
-|ResNet_Attention.py	#带有注意力机制的残差卷积模型
 |
-|ResAttention.py	#穿插注意力机制的网络模型
 |
-|transformer2.py #transformer模型
 |
-|train*.py		#训练并测试
 |
-|run*.py		#用户调用接口
 |
-|cnvkit_run.py	#基于拷贝数变异识别eccDNA方法
 |
-|verification.py	#验证准确度
 |
ps:*指有多个文件或多个版本编号，分别用于不同模型
``` 

## 环境配置
### 功能简述
WGS数据预处理(count*.py、SamtoolBash.sh)<br />
模型训练(dataprocess.py、dataloader*.py、train_attention.py)<br />
模型验证(verification.py)<br />
模型调用接口(run.py)<br />
基于CNV的McroDNA提取(cnvkit_run.py)<br />

### 配置命令
(cuda>11.7,或使用其他版本pytorch)<br />
``` 
conda/source activate/creat YOUR_ENV_NAME
pip install -r requirements.txt
``` 
## 模型使用
### 从自定义的长片段基因中识别MicroDNA
#### run文件运行命令示例
#快速识别./identify/long_segment_to_identify文件夹下的所有.fa序列文件<br /><br />
``` 
conda activate pytorch<br />
cd YOUR_DIR_PATH<br />
python run.py --pattern long_segment <br />
``` 
#python run.py输入参数<br />
--pattern(必选)，指定运行模式，短序列or长序列，InputType:short_sequence, long_segment<br />
--model(可选)，指定模型名称，模型文件在'./save/'文件夹下<br />
--file_path(可选)，指定文件所在文件夹，未指定时使用默认文件夹下文件<br />
--manual_input(可选)，手动输入序列<br />
--limit(可选)，灵敏阈值，默认为0.9<br />
#run文件需要提供模型文件，在.\save中<br />
#训练好的模型参数放在save文件中，也可以通过命令指定<br />
#默认存放fa文件的位置为./identify中两个文件夹，也可以自行指定<br />
#./identify/long_segment_to_identify	中可以放入fa文件，自动<br />读取所有fa文件并识别其中的MicroDNA，结果文件也存放在本文件夹下<br />
#./identify/fatsa_to_identify	短序列识别模式的结果文件<br />

## 模型训练
### 数据预处理
基于NCBI实验Series GSE68644、Series GSE124470<br />
https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE68644<br />
https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE124470<br />
下载GSE68644_RAW文件，从中读取eccDNA序列位置<br />
使用count*.py切割获取eccDNA序列<br />
使用cout_other*.py切割获取otherDNA序列<br />
#标准序列，GSE68644需要hg19参考序列，GSE124470需要hg38参考序列，放在运行路径中<br />
具体调用方式可在对应py文件中修改<br />

### 训练命令
``` 
conda activate pytorch
cd dir
python train*.py
``` 
#train.py文件需调用dataloader*.py、dataprocess*.py、模型文件<br />
#训练数据放在datasets中<br />

#### 使用tensorboard分析训练结果，在./run文件夹中打开cmd
``` 
tensorboard --logdir=// --port 8130
``` 

#### 使用verification.py验证结果
基于train.py文件开发，使用方式相同，将测试数据放入datasets<br />
运行py文件即可<br />
``` 
python verification.py
``` 
#### 绘制ROC图形，使用ROC_draw.py
训练后的ROC数据在ROC文件夹下<br />
直接运行py文件即可<br />
``` 
python ROC_draw.py
``` 
## 基于CNVs区域识别MicroDNA
#### 数据需要
WGS测序数据文件(可为SRA文件.sra或fastq文件.fq)、hg19的标准序列<br />
hg19.fa(来源于NCBI)与CNVkit提供的比对标准文件<br />
hg19_cnvkit_filtered_ref.cnn(来源于CNVkit)需放在
MicroDNA_Hook\cnvkit\cnvkit_do文件夹下<br />

tips：CNVkit的调用在windows11系统和windows-linux子系统中测试有问题，请参阅cnvkit使用方法，计算后将result.call.cns文件放在 MicroDNA_Hook\cnvkit_do\out中继续运行cnvkit_run.py即可。<br />

#### 基于CNVs区域识别MicroDNA运行命令
``` 
python cnvkit_run.py
``` 
--model(可选)，指定模型名称，模型文件在'./save/'文件夹下，default module.pth。<br />
--run(可选)，指定调用的版本，default run_v11.2.py。<br />
--limit(可选)，灵敏阈值，默认为0.95<br />


#### 合并结果用于您的后分析
``` 
python merge_file.py
``` 
##文件说明：glob遍历cnvkit_do\\fa下的所有.txt文件,写入cnvkit_do\connect文件夹下<br />

## 模型说明
1.ResNet_Attention.py	#带有注意力机制的残差卷积模型<br />
来自本科毕业论文"eccDNA Identification based on Deep Learning".配合module.pth使用。<br />
2.ResAttention.py	#穿插注意力机制的网络模型
性能更佳，配合6.pth使用，需在run.py文件中修改import代码<br />
3.经过测试transformer模型并不适用于本任务，可能由于数据量过少或单个数据的信息密度过低，在论文中有详细说明<br />
