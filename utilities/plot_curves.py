import argparse
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.metrics import (roc_curve, roc_auc_score, 
                             auc, precision_recall_curve)
from tensorflow.keras.utils import to_categorical


# multilabel processing not yet implemented


def make_roc(ground_truth, raw_preds, microavg=False):
    if microavg:
        # auroc = roc_auc_score(ground_truth, raw_preds, multi_class="ovr", average="micro")
        fpr, tpr, _ = roc_curve(ground_truth.ravel(), raw_preds.ravel())
        auroc = auc(fpr, tpr)
    else:
        auroc = roc_auc_score(ground_truth, raw_preds)
        fpr, tpr, _ = roc_curve(ground_truth, raw_preds)
    return auroc, fpr, tpr
        
    
def roc(ground_truth, raw_preds_list, output_dir, figsize=None, 
        att=False, model_names=["model"], source="source"):
    
    no_skill_outputs = np.ones(ground_truth.shape, dtype=np.float32)
    if figsize==None:
        figsize = plt.rcParams.get('figure.figsize')
    plt.figure(figsize=figsize)
        
    for i in range(len(raw_preds_list)+1):
        if i > 0:
            auroc, fpr, tpr = make_roc(ground_truth[0], raw_preds_list[i-1][0])
            plt.plot(fpr, tpr, linestyle="solid", marker='.', 
                     label=f"{model_names[i-1]} AUC: {auroc:.3f}")
        else:
            _, fpr, tpr = make_roc(ground_truth[0], no_skill_outputs[0])
            plt.plot(fpr, tpr, linestyle="dashed", label="No skill")
        
        plt.xlabel("False Positive Rate (FPR)")
        plt.xlim([-0.05,1.05])
        plt.ylabel("True Positive Rate (TPR) / Specificity")
        plt.ylim([-0.05,1.05])
        plt.legend(loc="lower right")
            
        if att:
            title = f"{source} Attribution ROC"
            fname = f"ATT_{source}_ROC.png"
        else:
            title = "Deepfake Detection ROC"
            fname = "DET_ROC.png"
        plt.title(title)
        
        plt.savefig(output_dir.joinpath(fname).__str__())
        print(f"Saved {fname} in {output_dir}.")
        plt.clf()
        
        
def make_prc(ground_truth, raw_preds, microavg=False):
    if microavg:
        precision, recall, _ = precision_recall_curve(
            ground_truth.ravel(), raw_preds.ravel())
    else:
        precision, recall, _ = precision_recall_curve(ground_truth, raw_preds)
    auprc = auc(recall, precision)
    return auprc, recall, precision
        
        
def prc(ground_truth, raw_preds_list, output_dir, figsize=None, 
        att=False, model_names=["model"], source="source"):
    
    if figsize==None:
        figsize = plt.rcParams.get('figure.figsize')
    plt.figure(figsize=figsize)
    
    for i in range(len(raw_preds_list)+1):
        if i > 0:
            auprc, recall, precision = make_prc(ground_truth[0], raw_preds_list[i-1][0])
            plt.plot(recall, precision, linestyle="solid", marker='.',
                     label=f"{model_names[i-1]} AUC: {auprc:.3f}")
        else:
            class_imbalance = len(ground_truth[0][ground_truth[0]==1]) / len(ground_truth[0])
            plt.plot([0, 1], [class_imbalance, class_imbalance], 
                     linestyle="dashed", label="No skill")
                
        plt.xlabel("Recall / Sensitivity")
        plt.xlim([-0.05,1.05])
        plt.ylabel("Precision")
        plt.ylim([-0.05,1.05])
        plt.margins(0.1)
        plt.legend()
            
        if att:
            title = f"{source} Attribution PRC"
            fname = f"ATT_{source}_PRC.png"
        else:
            title = "Deepfake Detection PRC"
            fname = "DET_PRC.png"
        plt.title(title)
        
        plt.savefig(output_dir.joinpath(fname).__str__())
        print(f"Saved {fname} in {output_dir}.")
        plt.clf()
        

def main(args):
    
    input_directory = Path(args.RESULTS_DIR.rstrip('/'))
    output_directory = (Path(args.output_dir) if args.output_dir else input_directory)
    ground_truth = None
    raw_preds = None
    attribution_mode = args.att
    
    for filepath in input_directory.iterdir():
        if filepath.name == "ground_truth.npy":
            ground_truth = np.load(filepath)
        elif filepath.name == "raw_predictions.npy":
            raw_preds = np.load(filepath)
        else:
            continue
        
    if (ground_truth is None) or (raw_preds is None):
        raise ValueError("Missing required input files!")
        
    if raw_preds.shape[0] > 1:
        attribution_mode = True
        ground_truth = np.expand_dims(ground_truth[1], axis=0)
        raw_preds = [np.expand_dims(raw_preds[1], axis=0)]
    else:
        raw_preds = [raw_preds]
        
    if not output_directory.exists():
        output_directory.mkdir(parents=True)
        
    if args.base_results:
        extra_directories = [Path(x.rstrip('/')) for x in args.base_results]
        extra_preds = [np.load(y.joinpath("raw_predictions.npy")) for y in extra_directories]
        if not extra_preds:
            raise ValueError("Missing required input files!")
        raw_preds.extend(extra_preds)
        if args.model_names:
            model_names = args.model_names
        else:
            model_names = [input_directory.parent.stem]
            model_names.extend([y.parent.stem for y in extra_directories])
    else:
        model_names = (args.model_names if args.model_names else [input_directory.parent.stem])

    if args.roc:
        roc(ground_truth, raw_preds, output_directory, att=attribution_mode, 
            model_names=model_names, source=args.source)
    if args.prc:
        prc(ground_truth, raw_preds, output_directory, att=attribution_mode, 
            model_names=model_names, source=args.source)
    

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("RESULTS_DIR",              help="Path to directory containing raw test output .npy files.", type=str)
    parser.add_argument("--base_results",   "-b",   help="Paths to directories containing .npy files for equivalent baseline test outputs.", nargs='*', type=str, default=[])
    parser.add_argument("--output_dir",     "-o",   help="Path to output directory. If not provided, save output in RESULTS_DIR.", type=str, default=None)
    parser.add_argument("--model_names",    "-m",   help="Identifiers of model instances being tested.", nargs='*', type=str, default=[])
    parser.add_argument("--roc",                    help="FLAG: output receiver operating characteristic curve.", action="store_true")
    parser.add_argument("--prc",                    help="FLAG: output precision-recall curve.", action="store_true")
    parser.add_argument("--att",            "-a",   help="FLAG: binary deepfake attribution mode.", action="store_true")
    parser.add_argument("--source",         "-l",   help="Designated source label for attribution.", type=str, default="source")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())