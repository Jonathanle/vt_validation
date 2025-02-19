""" Create a trainer that optimzes the model and allows for relevant properties to be shown
"""
import pdb
import torch
from torch.utils.data import Subset, DataLoader, WeightedRandomSampler
import torch.nn as nn

import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR, ExponentialLR

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from scoring_model import BaselineScorer, CNNScorer, RiskAssessmentModel, CNNScorerWithMasks
from dataset import Preprocessor, LGEDataset

from torch.utils.data import SubsetRandomSampler


from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
"""
Steps to validation 

1 Ensure a basic flow to optimzation --> ensure no blatant errors
2. once no blatant or critical errors --> verify the automated evaluation mechanism 
3. once this is verified, experiment with controlled approaches to impmrovement.


# based on how the model is behaving how should i optimize why do i do so? come up with reasons.

# i noticedd that the model performance is not learning --> this means that the complexity is not really doing anythig
nee dmore capability / decision making power for more complex decisions

- bottleneck 1 - informational content - actual metrics that we want.


here propose scientific method how to do --> peopose reasonable args about convergence and decision making.
keep stable finding and control factoors - think about outputs and represent., - samed as math - need simple models.
regularized finding
- most of what i do in deveoopment involves noise the ture skill here is evaluating precisely howt he model moves. 


- how do i decide? - dont the images have to be inputted within a specific manner ever time?

- think abt principles 0 get ds methods
- how does  a strat converge to 
"""

"""
Training Difficulties - Interpreting what does AUC actually mean? how cna I improve and see the results? can I control what I do in testing? 
- cognitvie is AUC being interpreted correctly? with probability values?o

- notes kfold cross - too noisy very hardd to implement - not 
- need a better way of evaluating systems of AI (better testing and representational systems)
- in ML and AI determine why it is soo hard to analyze and see waht methods I can do to diagnose specifici errors

AUC - useful for knowing if the model can actualy discriminate invariant to thresholds like precision

# issues in analyzing --> what is truely deterministic vs stochastic?

TODO: System to adequately visualize the matplotlib intermediate LGE segmentations + the actual LGE image emphasized for humans
"""


class TrainingConfig():
    def __init__(self): 
        self.learning_rate = 1e-4
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = 4 
        self.num_epochs = 10 

        self.pos_weight = 4.0 
        self.n_slices = 6 
class BalancedBCELoss(nn.Module):
    def __init__(self, beta=0.1):  # beta controls importance of positive class
        super().__init__()
        self.beta = beta
        
    def forward(self, pred, target):
        # Calculate weights for each class
        pos_weight = (1 - self.beta)/(1 - self.beta**torch.sum(target == 0))
        neg_weight = self.beta/self.beta**torch.sum(target == 1)
        
        # Apply weights to BCE loss
        weights = torch.where(target == 1, pos_weight, neg_weight)
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        weighted_bce = weights * bce
        
        return weighted_bce.mean()
class Trainer():
    """
    Class for Training LGE Model 


    Requirements: / TODO

    - init - TrainingConfig Object will tell how the optimization will behave. 
    - after every epoch trainer will call evaluator to calculate all relevant statistics
    - valildatiotn flow 
    - forward + backward optitmizer

    """
    # TODO: I broke my implementataion, i need to specifcy a better way of interfacign with both traijing and kfold
    # Idea dont hold trainers with specifc objects, rather use the methods as interfaces to take in the training and kfold.
    def __init__(self, model,evaluator, config): 

        self.model = model.to(config.DEVICE)

        self.config = config

        self.evaluator = evaluator

        self.optimizer = torch.optim.Adam(
            model.parameters(), 
            lr = config.learning_rate
        )
        pos_weight = torch.tensor([config.pos_weight]).to(config.DEVICE) # Error I need to push this into the devicd
        self.criterion = nn.BCEWithLogitsLoss(pos_weight = pos_weight)


    def train_epoch(self, loader):
        """Single training epoch."""
        #TODO: Evaluate to see if this correct and ideal for my kind of data
        self.model.train()
        total_loss = 0
        
        for batch_idx, (data, target, patient_id) in enumerate(loader): # things hard to analyze --> what are the dynamics of teh daataloader? 
            data = data.to(self.config.DEVICE)
            target = target.to(self.config.DEVICE)


            self.optimizer.zero_grad()
            
            
            output = self.model(data)


            # debugging step for examining accurayc of segmentations
            #self.model.plot_segmentations(patient_id, target)

            loss = self.criterion(output, target)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        return total_loss / len(loader)

    def save_checkpoint(self, is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_auc': self.best_val_auc
        }
        
        if is_best:
            path = self.config.checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, path)

    
    def train(self, train_dataloader, val_dataloader):
        """
        TODO: input the dataloader and val datalooader 
        """
        for epoch in range(self.config.num_epochs):
            # Training
            train_loss = self.train_epoch(train_dataloader)
            
            #self.scheduler.step()
            
            
            # Validation
            val_metrics = self.evaluator.evaluate(val_dataloader)
            val_auc = val_metrics['auc']
            sensitivity = val_metrics['sensitivity']
            specificity = val_metrics['specificity']
            precision = val_metrics['precision']
            recall = val_metrics['recall']

            """         
            # Learning rate scheduling
            self.scheduler.step(val_auc)
            
            # Save best model
            if val_auc > self.best_val_auc:
                self.best_val_auc = val_auc
                self.patience_counter = 0
                self.save_checkpoint(is_best=True)
            else:
                self.patience_counter += 1
            
            # Early stopping
            if self.patience_counter >= self.config.patience:
                print("Early stopping triggered")
                break
            
            """
            print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val AUC = {val_auc:.4f} precision: {precision:.4f} recall {recall} sens {sensitivity} spec {specificity}")



    def _reinitialize_model(self):
        """Method 1: Create a new instance of the model"""
        # Assuming self.model is a CNNScorerWithMasks
        new_model = CNNScorerWithMasks(n_slices=self.config.n_slices)
        new_model = new_model.to(self.config.DEVICE)
        self.model = new_model
        

        self.evaluator = Evaluator(self.model, self.config.DEVICE)
        # Reinitialize optimizer with new model parameters
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate
        )
    def train_k_fold(self, k_fold_loaders):
        """Main training loop with validation. Uses kfold for validating"""

        metrics = np.zeros((len(k_fold_loaders), 5), dtype=np.float32)

        for i, (train_loader, val_loader) in enumerate(k_fold_loaders): 
            

            self._reinitialize_model()
            for epoch in range(self.config.num_epochs):
                # Training
                train_loss = self.train_epoch(train_loader)
                
                # Validation
                val_metrics = self.evaluator.evaluate(val_loader)
                val_auc = val_metrics['auc']
                sensitivity = val_metrics['sensitivity']
                specificity = val_metrics['specificity']
                precision = val_metrics['precision']
                recall = val_metrics['recall']

                metrics[i] = np.array([val_auc, sensitivity, specificity, precision, recall])

                print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val AUC = {val_auc:.4f} precision: {precision:.4f} recall {recall} sens {sensitivity} spec {specificity}")
        scores = np.mean(metrics, axis = 0)
        print(f"auc: {scores[0]} sens: {scores[1]} spec: {scores[2]} precision: {scores[3]}")


class Evaluator():
    """Core evaluator with essential metrics."""
    def __init__(self, model, device):
        self.model = model
        self.device = device
    
    @torch.no_grad()
    def evaluate(self, data_loader):
        """Evaluate model with core metrics."""
        self.model.eval()



        predictions, targets, patient_ids = self._get_predictions(data_loader)
        

        auc, accuracy, true_positives, false_negatives, true_negatives, false_positives = self._get_metrics(predictions, targets)
        #pdb.set_trace()


        # Calculate more comples metrics (This one is easier to verify)


        # Verifying the modularization of the behavior
        print(f"tp: {true_positives} false_negatives: {false_negatives} tn: {true_negatives} fp {false_positives}")
        
        sensitivity = true_positives / (true_positives + false_negatives)
        specificity = true_negatives / (true_negatives + false_positives)


        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / (true_positives + false_negatives)
        return {
            'auc': auc,
            'accuracy': accuracy,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'recall': recall,
            'precision': precision
        }


    def _get_predictions(self, data_loader):
        predictions = []
        targets = []
        patient_ids = []
        for data, target, patient_id in data_loader:
            data = data.to(self.device)
            output = self.model(data)
            pred = torch.sigmoid(output)


            predictions.extend(pred.detach().cpu().numpy())
            targets.extend(target.cpu().numpy())
            patient_ids.extend(patient_id.cpu().numpy())
        # at this point we can get and parse this behavior more rigorousl        

        # Are these specifically np array of shape what? these 1d arrays
        return np.array(predictions), np.array(targets), np.array(patient_ids)

    def _get_metrics(self, predictions, targets):

        "internal methods are tested for helpers, underscores emphasize that we do NOT use these functions externally, but allow for testing"

        """
        calcualtes the predictions and targets given np array 
        """        
        # Calculate core metrics

        auc = roc_auc_score(targets, predictions)

        iplot = self.plot_roc_auc(targets, predictions)

        accuracy = ((predictions > 0.5) == targets).mean()
        
        # Calculate sensitivity and specificity
        true_positives = ((predictions > 0.5) & (targets == 1)).sum()
        false_negatives = ((predictions <= 0.5) & (targets == 1)).sum()
        true_negatives = ((predictions <= 0.5) & (targets == 0)).sum()
        false_positives = ((predictions > 0.5) & (targets == 0)).sum()

        return auc, accuracy, true_positives, false_negatives, true_negatives, false_positives

    def plot_roc_auc(self, y_true, y_pred_proba):

        # Calculate ROC curve and ROC area
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        # Plot
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True)
        return plt

    # Usage example:
    # plot_roc_auc(y_true, model.predict_proba(X_test)[:, 1])
    # plt.show()

class DatasetSplitter(): 
    def __init__(self): 
        pass
    def split_dataset(self, dataset, combine_train_validation_sets = False):
        seed = 52 

        indices = list(range(len(dataset)))
        labels = [dataset[i][1] for i in indices]

        train_idx, test_idx = train_test_split(
            indices,
            stratify = labels, 
            test_size = 0.3,
            random_state = seed, # generate the ROC - Curve -- Intermediate - 
        )


        test_labels = [dataset[i][1] for i in test_idx]


        val_idx, final_test_idx = train_test_split(
            test_idx,
            stratify = test_labels, 
            test_size = 0.5,
            random_state=seed
        )

        if combine_train_validation_sets: 

            train_dataset = Subset(dataset, train_idx + val_idx)
            val_dataset = None 
        else:
            train_dataset = Subset(dataset, train_idx)
            val_dataset = Subset(dataset, val_idx)

        test_dataset = Subset(dataset, final_test_idx)

        return train_dataset, val_dataset, test_dataset
    def create_fold_dataloaders(self, dataset, n_splits=3, batch_size=32, random_state=42):
        """
        Creates DataLoaders for each fold in stratified k-fold cross validation.
        
        Returns:
            list of tuples (train_loader, val_loader) for each fold
        """
        # Get labels for stratification
        labels = [dataset[i][1] for i in range(len(dataset))]
        
        # Initialize StratifiedKFold
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        
        # Create DataLoaders for each fold
        fold_loaders = []
        
        for train_idx, val_idx in skf.split(np.zeros(len(dataset)), labels):
            train_sampler = SubsetRandomSampler(train_idx)
            val_sampler = SubsetRandomSampler(val_idx)
            
            train_loader = DataLoader(
                dataset,
                batch_size=batch_size,
                sampler=train_sampler,
                num_workers=2,
                pin_memory=True
            )
            
            val_loader = DataLoader(
                dataset,
                batch_size=batch_size,
                sampler=val_sampler,
                num_workers=2,
                pin_memory=True
            )
            
            fold_loaders.append((train_loader, val_loader))
        
        return fold_loaders
def create_weighted_sampler(dataset, labels):
    # Calculate class weights
    class_counts = np.bincount(labels)
    class_weights = 1. / class_counts
    
    # Assign weight to each sample
    sample_weights = class_weights[labels]
    
    # Create sampler
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(labels),
        replacement=True
    )
    return sampler



def main():
    config = TrainingConfig()


    # Code Requires ./dataprocessed mri_data repository
    pp = Preprocessor()
    X, y, ids = pp.transform(config.n_slices, include_masks= True) # this dataset 

    dataset = LGEDataset(X, y, ids)

    dataset_splitter = DatasetSplitter()
    train_dataset, _ , test_dataset = dataset_splitter.split_dataset(dataset, combine_train_validation_sets=True)


    # confusing logic for dataset splitting

    td2, vd2, _ = dataset_splitter.split_dataset(dataset, combine_train_validation_sets=False)

    train_dataloader = DataLoader(train_dataset, batch_size = config.batch_size)
   # val_dataloader = DataLoader(val_dataset, batch_size = config.batch_size)


    labels = [dataset[i][1] for i in range(len(td2))]

    weights = [1e-10 if label == 0 else 1e10 for label in labels]
    sampler = WeightedRandomSampler(weights = weights, num_samples = len(weights), replacement = False)



    tdl2 = DataLoader(td2, batch_size = config.batch_size, sampler = sampler)

    batch = next(iter(tdl2))
    vdl2 = DataLoader(vd2, batch_size = config.batch_size)

    k_fold_loaders = dataset_splitter.create_fold_dataloaders(train_dataset)


    model = CNNScorerWithMasks(n_slices = config.n_slices)
    evaluator = Evaluator(model, config.DEVICE)

    trainer = Trainer(model,  evaluator, config)

    #trainer.train_k_fold(k_fold_loaders) # here we don't need to be tied to a specific dataset and let it vary
    trainer.train(tdl2, vdl2)
    return



if __name__ == '__main__':
   main() 