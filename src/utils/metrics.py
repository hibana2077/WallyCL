import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix, classification_report
from typing import List, Dict, Tuple, Optional
import torch


class WallyClMetrics:
    """Metrics computation for WallyCL training and evaluation"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset accumulated metrics"""
        self.predictions = []
        self.targets = []
        self.odd_predictions = []
        self.odd_targets = []
    
    def update_classification(self, predictions: List[int], targets: List[int]):
        """Update classification metrics"""
        self.predictions.extend(predictions)
        self.targets.extend(targets)
    
    def update_odd_detection(self, odd_predictions: List[int], odd_targets: List[int]):
        """Update odd-one-out detection metrics"""
        self.odd_predictions.extend(odd_predictions)
        self.odd_targets.extend(odd_targets)
    
    def compute_classification_metrics(self, predictions: Optional[List[int]] = None, 
                                     targets: Optional[List[int]] = None) -> Dict[str, float]:
        """Compute classification metrics"""
        if predictions is None:
            predictions = self.predictions
        if targets is None:
            targets = self.targets
        
        if not predictions or not targets:
            return {'accuracy': 0.0, 'f1_macro': 0.0, 'f1_micro': 0.0, 
                   'precision': 0.0, 'recall': 0.0}
        
        predictions = np.array(predictions)
        targets = np.array(targets)
        
        metrics = {
            'accuracy': accuracy_score(targets, predictions),
            'f1_macro': f1_score(targets, predictions, average='macro', zero_division=0),
            'f1_micro': f1_score(targets, predictions, average='micro', zero_division=0),
            'precision': precision_score(targets, predictions, average='macro', zero_division=0),
            'recall': recall_score(targets, predictions, average='macro', zero_division=0)
        }
        
        return metrics
    
    def compute_odd_detection_metrics(self, odd_predictions: Optional[List[int]] = None,
                                    odd_targets: Optional[List[int]] = None) -> Dict[str, float]:
        """Compute odd-one-out detection metrics"""
        if odd_predictions is None:
            odd_predictions = self.odd_predictions
        if odd_targets is None:
            odd_targets = self.odd_targets
        
        if not odd_predictions or not odd_targets:
            return {'odd_accuracy': 0.0}
        
        odd_predictions = np.array(odd_predictions)
        odd_targets = np.array(odd_targets)
        
        return {
            'odd_accuracy': accuracy_score(odd_targets, odd_predictions)
        }
    
    def compute_top_k_accuracy(self, logits: torch.Tensor, targets: torch.Tensor, 
                              k: int = 5) -> float:
        """Compute top-k accuracy"""
        if logits.size(0) == 0:
            return 0.0
        
        _, top_k_pred = torch.topk(logits, k, dim=1)
        targets = targets.view(-1, 1).expand_as(top_k_pred)
        correct = top_k_pred.eq(targets).float().sum(dim=1)
        
        return correct.mean().item()
    
    def compute_confusion_matrix(self, predictions: Optional[List[int]] = None,
                               targets: Optional[List[int]] = None,
                               class_names: Optional[List[str]] = None) -> np.ndarray:
        """Compute confusion matrix"""
        if predictions is None:
            predictions = self.predictions
        if targets is None:
            targets = self.targets
        
        if not predictions or not targets:
            return np.array([[]])
        
        cm = confusion_matrix(targets, predictions)
        return cm
    
    def get_classification_report(self, predictions: Optional[List[int]] = None,
                                targets: Optional[List[int]] = None,
                                class_names: Optional[List[str]] = None) -> str:
        """Get detailed classification report"""
        if predictions is None:
            predictions = self.predictions
        if targets is None:
            targets = self.targets
        
        if not predictions or not targets:
            return "No predictions available"
        
        return classification_report(
            targets, predictions, 
            target_names=class_names,
            zero_division=0
        )


class DistributionMetrics:
    """Metrics for analyzing data distribution and model predictions"""
    
    @staticmethod
    def compute_class_distribution(targets: List[int]) -> Dict[int, int]:
        """Compute class distribution"""
        from collections import Counter
        return dict(Counter(targets))
    
    @staticmethod
    def compute_prediction_confidence(logits: torch.Tensor) -> Dict[str, float]:
        """Compute prediction confidence statistics"""
        probs = torch.softmax(logits, dim=1)
        max_probs = torch.max(probs, dim=1)[0]
        
        return {
            'mean_confidence': max_probs.mean().item(),
            'std_confidence': max_probs.std().item(),
            'min_confidence': max_probs.min().item(),
            'max_confidence': max_probs.max().item()
        }
    
    @staticmethod
    def compute_entropy(logits: torch.Tensor) -> Dict[str, float]:
        """Compute prediction entropy"""
        probs = torch.softmax(logits, dim=1)
        log_probs = torch.log(probs + 1e-8)
        entropy = -(probs * log_probs).sum(dim=1)
        
        return {
            'mean_entropy': entropy.mean().item(),
            'std_entropy': entropy.std().item(),
            'min_entropy': entropy.min().item(),
            'max_entropy': entropy.max().item()
        }


class CalibrationMetrics:
    """Metrics for model calibration assessment"""
    
    @staticmethod
    def compute_ece(logits: torch.Tensor, targets: torch.Tensor, 
                   n_bins: int = 15) -> float:
        """
        Compute Expected Calibration Error (ECE)
        """
        probs = torch.softmax(logits, dim=1)
        confidences, predictions = torch.max(probs, dim=1)
        accuracies = predictions.eq(targets)
        
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0.0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = confidences.gt(bin_lower.item()) & confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece.item()
    
    @staticmethod
    def compute_reliability_diagram(logits: torch.Tensor, targets: torch.Tensor,
                                  n_bins: int = 15) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute data for reliability diagram
        Returns: (bin_centers, accuracies, confidences)
        """
        probs = torch.softmax(logits, dim=1)
        confidences, predictions = torch.max(probs, dim=1)
        accuracies = predictions.eq(targets)
        
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2
        
        bin_accuracies = []
        bin_confidences = []
        
        for i in range(n_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]
            
            in_bin = confidences.gt(bin_lower.item()) & confidences.le(bin_upper.item())
            
            if in_bin.sum() > 0:
                bin_acc = accuracies[in_bin].float().mean().item()
                bin_conf = confidences[in_bin].mean().item()
            else:
                bin_acc = 0.0
                bin_conf = 0.0
            
            bin_accuracies.append(bin_acc)
            bin_confidences.append(bin_conf)
        
        return bin_centers.numpy(), np.array(bin_accuracies), np.array(bin_confidences)
