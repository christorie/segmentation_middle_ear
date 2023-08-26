import numpy as np
import os
import matplotlib.pyplot as plt
import json
import sys
import os
import re


if __name__ == "__main__":
    if len(sys.argv) > 1:
        model = sys.argv[1]
        print("Model:", model)

        # create variable paths
        modelpath = f'%PATH%\\radiology\\model\\segmentation_middle_ear\\{ model }'
        mlpath = f'{ modelpath }\\mlruns'
        regex_stats = re.compile('stats_*.*_*.*.json')
        for root, dirs, files in os.walk(modelpath):
            for file in files:
                if regex_stats.match(file):
                    train_js = os.path.join(modelpath, file) 
        # print(train_js)
        # e.g. %PATH%\radiology\model\segmentation_middel_ear\train_08\stats_20230416_165017.json

        metrics = 'metrics'
        for root, dirs, files in os.walk(mlpath):
            if metrics in dirs:
                metrics_path=(os.path.join(root, metrics))
        loss_file = os.path.join(metrics_path, "loss")
        #print(loss_file)
        # e.g. %PATH%\radiology\model\segmentation_middle_ear\train_08\mlruns\769768581866760050\9cd22fb190ee47e883bf6db7b91d2e6b\metrics\loss


        # get amount of epochs and iterations from train_stats.json 
        with open(train_js) as json_file:
            data = json.load(json_file)
            epoch = data['epoch']
            iterate = data['total_iterations']


        # read loss file, append to loss_info
        loss_info = []
        with open(loss_file) as f:
            [loss_info.append(line) for line in f.readlines()]


        # create lists 
        loss = []
        iterations = list(range(1, iterate+1)) # 1 to iterations
        epochs = list(range(1, epoch+1)) # 1 to epochs
        epoch_loss = []


        # get all loss values (for each iteration)
        for row in range(len(loss_info)):
            loss.append(round(float(loss_info[row].split()[1]), 3))


        # get all loss values (for each epoch)
        for epoch_iter in range(len(loss)+1):
            if epoch_iter % iterate == 0 and epoch_iter != 0:
                epoch_loss.append(loss[epoch_iter-1]) #list of iterations that end an epoch
        #print(epoch_loss)


        # plot all epoch losses over epochs:
        plt.plot(epochs, epoch_loss)
        plt.xlabel('epochs')
        plt.xticks(np.arange(0, len(epochs)+1, 10))  
        plt.ylabel('loss')
        plt.yticks(np.arange(0, 2.05, 0.05)) 
        # plt.legend()
        plt.grid(visible=True, which='major', axis='both')
        plt.title('Dice Loss (Train)', fontdict=None, loc='center', pad=None)
        plt.show()


    else:
        print("No model selected.")

    
