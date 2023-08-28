# Created by Hansi at 28/08/2023
from seqeval.metrics import classification_report, precision_score, recall_score, f1_score
from seqeval.scheme import IOB2


def print_eval_ner(actuals, preds, eval_file_path=None):
    f = None
    if eval_file_path is not None:
        f = open(eval_file_path, "w")

    cls_report = classification_report(actuals, preds)
    print("Default classification report:\n")
    print(cls_report)
    if eval_file_path is not None:
        f.write("Default classification report:\n")
        f.write("{}\n".format(cls_report))

    cls_report_strict = classification_report(actuals, preds, mode="strict", scheme=IOB2)
    print("Strict classification report:\n")
    print(cls_report_strict)

    if eval_file_path is not None:
        f.write("Strict classification report:\n")
        f.write("{}\n".format(cls_report_strict))

    result = {
        "precision(macro)": precision_score(actuals, preds, average="macro"),
        "recall(macro)": recall_score(actuals, preds, average="macro"),
        "f1_score(macro)": f1_score(actuals, preds, average="macro"),
        "precision_strict(macro)": precision_score(actuals, preds, average="macro", mode="strict", scheme=IOB2),
        "recall_strict(macro)": recall_score(actuals, preds, average="macro", mode="strict", scheme=IOB2),
        "f1_score_strict(macro)": f1_score(actuals, preds, average="macro", mode="strict", scheme=IOB2)
    }

    for key in result.keys():
        print(f'{key} = {result[key]}')
        if eval_file_path is not None:
            f.write(f"{key} = {str(result[key])}\n")

    if eval_file_path is not None:
        f.close()
