import matplotlib.pyplot as plt
import glob

all_ROC_path = glob.glob(r'./ROC/*.txt')
for ROC in all_ROC_path:

    draw = []
    with open(ROC, "r", encoding='utf-8') as ROC_COM:
        datas = ROC_COM.readlines()
        for data in datas:
            Line=data.strip().split("   ",2) 
            draw.append(Line)
        #print(draw)

        x = []
        y = []
        for i in draw:
            y.append(float(i[0]))
            x.append(float(i[1]))
        #print(x,'\n',y)            ###test

        ###计算ROC面积
        area = 0
        nn = 0  ###计数器
        long = len(x)
        for nn in range(long-1):
            area = area+(x[nn+1]-x[nn])*(y[nn+1]+y[nn])/2


        word = 'AUC: '+str('{:.3f}'.format(area))
        
        plt.plot(x,y)
        plt.text(0.6, 0.5, word)
        plt.savefig(ROC[:-4]+'.png')
        plt.cla()
        plt.close()

        #plt.show()
