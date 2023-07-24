# Created by Hansi at 24/07/2023
from sklearn.metrics import f1_score, recall_score, precision_score, classification_report


def macro_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average='macro')


def macro_recall(y_true, y_pred):
    return recall_score(y_true, y_pred, average='macro')


def macro_precision(y_true, y_pred):
    return precision_score(y_true, y_pred, average='macro')


def print_eval_re(actuals, preds, eval_file_path=None):
    f = None
    if eval_file_path is not None:
        f = open(eval_file_path, "w")

    cls_report = classification_report(actuals, preds)
    print("Classification report:\n")
    print(cls_report)
    if eval_file_path is not None:
        f.write("Default classification report:\n")
        f.write("{}\n".format(cls_report))

    result = {
        "precision(macro)": precision_score(actuals, preds, average="macro"),
        "recall(macro)": recall_score(actuals, preds, average="macro"),
        "f1_score(macro)": f1_score(actuals, preds, average="macro"),
    }

    for key in result.keys():
        print(f'{key} = {result[key]}')
        if eval_file_path is not None:
            f.write(f"{key} = {str(result[key])}\n")

    if eval_file_path is not None:
        f.close()