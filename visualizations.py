import matplotlib.pyplot as plt


class Plot():
    def __init__(self, results):
        self.results = results

        self.visualize(metric1="tr_iou", 
                       metric2="val_iou", 
                       label1="Train IoU",
                       label2 ="Validation IoU", 
                       title="Mean Intersection Over Union Learning Curve", 
                       ylabel="mIoU Score")

        self.visualize(metric1="tr_pa", 
                       metric2="val_pa", 
                       label1="Train PA",
                       label2="Validation PA", 
                       title="Pixel Accuracy Learning Curve", 
                       ylabel="PA Score")

        self.visualize(metric1="tr_loss", 
                       metric2="val_loss", 
                       label1="Train Loss",
                       label2="Validation Loss", 
                       title="Loss Learning Curve", 
                       ylabel="Loss Value")

    def plot(self, metric, label): 
        plt.plot(self.results[metric], label=label)

    def decorate(self, ylabel, title): 
        plt.title(title)
        plt.xlabel("Epochs")
        plt.ylabel(ylabel)
        plt.legend()
        plt.show()

    def visualize(self, metric1, metric2, label1, label2, title, ylabel):
        plt.figure(figsize=(10, 5))
        self.plot(metric1, label1)
        self.plot(metric2, label2)
        self.decorate(ylabel, title)