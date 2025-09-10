import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, ConfusionMatrixDisplay, classification_report

# Tester Function for Categorical Predictions
class Tester:

    def __init__(self, predictor, data, title=None, size=250):
        self.predictor = predictor
        self.data = data
        self.title = title or predictor.__name__.replace("_", " ").title()
        self.size = size
        self.guesses = []
        self.truths = []

    def run_datapoint(self, i):
        datapoint = self.data[i]
        guess = self.predictor(datapoint)
        truth = datapoint.label   # assuming `label` is the categorical ground truth
        self.guesses.append(guess)
        self.truths.append(truth)
        title = datapoint.text if len(datapoint.text) <= 40 else datapoint.text[:40]+"..."
        print(f"{i+1}: Guess={guess} Truth={truth} Item: {title}")

    def chart_confusion_matrix(self):
        cm = confusion_matrix(self.truths, self.guesses, labels=[0,1,2])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0,1,2])
        disp.plot(cmap=plt.cm.Blues)
        plt.title(f"{self.title} - Confusion Matrix")
        plt.show()

    def report(self):
        acc = accuracy_score(self.truths, self.guesses)
        precision, recall, f1, _ = precision_recall_fscore_support(self.truths, self.guesses, average="weighted")
        
        print(f"\n===== {self.title} Report =====")
        print(f"Accuracy: {acc:.2f}")
        print(f"Precision: {precision:.2f}")
        print(f"Recall: {recall:.2f}")
        print(f"F1 Score: {f1:.2f}")
        
        self.chart_confusion_matrix()

    def run(self):
        for i in range(self.size):
            self.run_datapoint(i)
        self.report()

    @classmethod
    def test(cls, function, data):
        cls(function, data).run()