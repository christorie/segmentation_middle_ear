import numpy as np
import os
import matplotlib.pyplot as plt
import sys


if __name__ == "__main__":
    if len(sys.argv) > 1:
        model = sys.argv[1]
        print("Model:", model)
        

        # create variable paths
        modelpath = f'%PATH%\\radiology\\model\\segmentation_middle_ear\\{ model }'
        mlpath = f'{ modelpath }\\mlruns'
        metrics = 'metrics'
        for root, dirs, files in os.walk(mlpath):
            if metrics in dirs:
                metrics_path=(os.path.join(root, metrics))
        

        # vars for different mean dice objects
        mean_dice_ce = os.path.join(metrics_path, "val_mean_dice")
        #md_ossicle_chain = os.path.join(metrics_path, "val_ossicle chain_mean_dice")
        #md_tympanic_cavity = os.path.join(metrics_path, "val_tympanic cavity_mean_dice")


        # get mean dice info
        md = []
        with open(mean_dice_ce) as f:
            [md.append(line) for line in f.readlines()]
        #print(data)


        # get mean dice and epochs
        mean_dice = []
        epoch = []
        for row in range(len(md)):
            mean_dice.append(round(float(md[row].split()[1]), 3))
            epoch.append(int(md[row].split()[2]))
        #print(mean_dice)
        #print(epoch)


        # plot mean dice over epochs:
        plt.plot(epoch, mean_dice)
        plt.xlabel('epochs')
        plt.xticks(np.arange(0, max(epoch)+1, 10)) 
        plt.ylabel('mean_dice')
        plt.yticks(np.arange(0, 1.05, 0.05)) 
        # plt.legend()
        plt.grid(visible=True, which='major', axis='both')
        plt.title('Mean Dice (Validation)', fontdict=None, loc='center', pad=None)
        plt.show()

        
    else:
        print("No model selected.")

    

