import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import torch

def confusion_matrix(num_classes, model, dataset):
    matrix = torch.zeros(num_classes, num_classes)
    df = pd.DataFrame(matrix, index = dataset.mapping, columns = dataset.mapping)
    dataloader = dataset.test_dataloader()
    for batch in dataloader:
        x, y = batch[0].to("cuda"), batch[1]
        preds = model.predict_step(x)
        preds = torch.argmax(preds, dim = 1)
        
        for i in range(len(preds)):
            df.iloc[y[i].item(), preds[i].item()] += 1
        
    plt.figure(figsize = (10, 8))
    plt.tick_params(axis='both', which='major', labelsize=10, labelbottom = False, bottom=False, top = False, labeltop=True)
    sns.heatmap(df, annot = True, fmt = ".1f")