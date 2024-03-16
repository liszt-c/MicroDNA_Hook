# MicroDNA Hook

### Folder structure

``` 
-|datasets- |eccDNA	    #Insert the eccDNA sequence, .fa
 |		   |
 | 		   |otherDNA	#Put in other DNA sequences, .fa
 |
-|cnvkit_do- |out	#cnvkit output file
 |		    |
 | 		    |fa	# stored intermediate and result file 
 |
-|identify- |fatsa_to_identify	#Short sequence identification method default folder
 |		   |
 | 		   |long_segment_to_identify #Long sequence identification method default folder
 |
 |ROC			#ROC output and image drawing folder
 |
-|run			#Training intermediate parameter storage folder
 |
-|save		    #Model storage folder

### File
-|preprocessing-	|SamtoolBash.sh	#The bash command provides the samtools interface
 |			       |
 |			       |count*.py		#The xcel documentation is read and SamtoolBash.sh is called to cut the eccDNA sequence
 |			       |
 |			       |cout_other*.py	#Read the xcel documentation and call SamtoolBash.sh to cut other DNA sequences
 |
-|dataprocess*.py	#The DNA sequence was read and the sequence was converted to a matrix
 |
-|dataloader*.py	#The transformed matrices were read, datasets and dataloader were constructed by pytorch, and 20% were used as the test set
 |
-|ResNet_Attention.py	#Residual convolution model with attention mechanism
 |
-|ResAttention.py	#Network models of interspersed attention mechanisms
 |
-|transformer2.py #transformer model
 |
-|train*.py		#Train and test
 |
-|run*.py		#User invocation interface
 |
-|cnvkit_run.py	#Methods for identification of eccDNA based on copy number variation
 |
-|verification.py	#Verification of accuracy
 |
ps: * Refers to having multiple files or version numbers for different models
```
## Environment Configuration
### Function Overview
WGS data preprocessing (count*.py, SamtoolBash.sh)<br /> Model training (dataprocess.py, dataloader*.py, train_attention.py)<br /> Model validation (verification.py)<br /> Model invocation API (run.py)<br /> Extraction of MicroDNA based on CNVs (cnvkit_run.py)
### Configuration Commands
CUDA version > 11.7, or use another version of PyTorch<br />
``` 
conda/source activate/creat YOUR_ENV_NAME
pip install -r requirements.txt
``` 
## Model Usage
### Identifying MicroDNA from Custom Long Segments
#### run.py command examples
#Rapid identification of all .fa sequence files in the ./identify/long_segment_to_identify folder<br />
``` 
conda activate pytorch
cd YOUR_DIR_PATH
python run.py --pattern long_segment 
``` 
#### python run.py input parameters<br />
--pattern (required), specifies the running mode, short sequence or long segment, InputTypes: short_sequence, long_segment<br /> --model (optional), specifies the model name; model files are located in the './save/' folder<br /> --file_path (optional), specifies the directory containing files; if not specified, default folder's files will be used<br /> --manual_input (optional), manually input sequences<br /> --limit (optional), sensitivity threshold, default is 0.9<br />

#The run.py file requires model files to be provided in .\save\ <br />
#Trained model parameters are stored in the save folder and can also be specified via command line<br />
#Default location for fa files is ./identify with two folders; this can be customized<br />
#./identify/long_segment_to_identify - Place fa files here to automatically read and identify MicroDNA; results are saved in this folder<br />
#./identify/fasta_to_identify - Folder for short sequence recognition mode result files<br />

## Model Training
### Data Preprocessing
Based on NCBI experiments Series GSE68644, Series GSE124470<br /> https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE68644<br /> https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE124470<br /> Download GSE68644_RAW files and extract eccDNA sequence positions<br /> Use count*.py to obtain eccDNA sequences<br /> Use cout_other*.py to obtain otherDNA sequences<br />

#Standard sequences: GSE68644 requires hg19 reference sequence, GSE124470 requires hg38 reference sequence, place them in the runtime path<br />
Modification details can be found within the corresponding py files<br />

### Training Commands
``` 
conda activate pytorch
cd dir
python train*.py
``` 
#train.py needs to invoke dataloader*.py, dataprocess*.py, and model files<br />
#Training data is placed in the datasets directory<br />

#### Analyzing training results using TensorBoard in the ./run folder
``` 
tensorboard --logdir=// --port 8130
``` 

#### Verifying results with verification.py
Developed based on train.py; place test data in the datasets directory<br /> Run the python file directly<br />
``` 
python verification.py
``` 
#### Drawing ROC curves using ROC_draw.py
Post-training ROC data is located in the ROC folder<br /> Run the python file directly<br />
``` 
python ROC_draw.py
``` 
## MicroDNA Identification Based on CNVs Regions
#### Required Data
WGS sequencing data files (can be SRA files .sra or fastq files .fq), hg19 standard sequence<br /> hg19.fa (from NCBI) and alignment reference files provided by CNVkit<br /> hg19_cnvkit_filtered_ref.cnn (from CNVkit) should be placed in the MicroDNA_Hook\cnvkit\cnvkit_do folder<br />

tip: CNVkit may have issues when invoked in Windows 11 systems or Windows Subsystem for Linux. Please refer to CNVkit usage instructions. After calculation, place the resulting result.call.cns file in MicroDNA_Hook\cnvkit_do\out and continue to run cnvkit_run.py.<br />

#### Command to Identify MicroDNA in CNVs Regions
``` 
python cnvkit_run.py
``` 
--model (optional), specifies the model name; model files are located in './save/', default is module.pth.<br /> --run (optional), specifies the version to be called, default is run_v11.2.py.<br /> --limit (optional), sensitivity threshold, default is 0.95<br />

#### Merging Results for Further Analysis
``` 
python merge_file.py
``` 
#File Description: glob traverses all .txt files in cnvkit_do\fa, writes into cnvkit_do\connect folder<br />

## Model Description
1. ResNet_Attention.py # Residual convolutional model with attention mechanism<br /> Used in the undergraduate thesis "eccDNA Identification based on Deep Learning"; works with module.pth.<br />
2. ResAttention.py # Network model with interleaved attention mechanism<br /> Performs better and should be used with 6.pth; modify import code in run.py accordingly.<br />
3. Transformer models have been tested but are not suitable for this task due to either insufficient data volume or low information density per sample, as detailed in the paper.<br />
