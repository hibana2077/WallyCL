import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
from .token_utils import cosine_distance, euclidean_distance


class OddOneOutLoss(nn.Module):
    """Odd-One-Out identification loss for group-based learning"""
    
    def __init__(self, alpha: float = 5.0, distance_type: str = 'cosine'):
        super().__init__()
        self.alpha = alpha
        self.distance_fn = cosine_distance if distance_type == 'cosine' else euclidean_distance
    
    def forward(self, group_embeddings: torch.Tensor, odd_index: int) -> torch.Tensor:
        """
        Args:
            group_embeddings: [K, embed_dim] where K = m + 1
            odd_index: index of the odd sample (usually K-1)
        Returns:
            loss: scalar loss value
        """
        K = group_embeddings.size(0)
        m = K - 1  # number of positives
        
        # Compute inconsistency scores for each sample in the group
        scores = []
        for k in range(K):
            # Average distance to all positives (excluding odd sample)
            distances = []
            for i in range(m):  # positives are indices 0 to m-1
                if i != k:  # don't compare with itself if k is in positives
                    distances.append(self.distance_fn(group_embeddings[k], group_embeddings[i]))
            
            if distances:
                avg_distance = torch.stack(distances).mean()
            else:
                # Handle case where k is the only positive (shouldn't happen in practice)
                avg_distance = torch.tensor(0.0, device=group_embeddings.device)
            
            scores.append(avg_distance)
        
        scores = torch.stack(scores)
        
        # Apply temperature scaling and softmax
        scaled_scores = self.alpha * scores
        probabilities = F.softmax(scaled_scores, dim=0)
        
        # Cross-entropy loss with odd_index as target
        target = torch.tensor(odd_index, device=group_embeddings.device, dtype=torch.long)
        return F.cross_entropy(scaled_scores.unsqueeze(0), target.unsqueeze(0))


class GroupSupConLoss(nn.Module):
    """Multi-positive supervised contrastive loss for groups"""
    
    def __init__(self, tau: float = 0.1):
        super().__init__()
        self.tau = tau
    
    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings: [batch_size, embed_dim] normalized embeddings
            labels: [batch_size] class labels
        Returns:
            loss: scalar loss value
        """
        batch_size = embeddings.size(0)
        device = embeddings.device
        
        # Compute similarity matrix
        sim_matrix = torch.matmul(embeddings, embeddings.t()) / self.tau
        
        # Create masks for positives and negatives
        labels = labels.contiguous().view(-1, 1)
        mask_pos = torch.eq(labels, labels.t()).float().to(device)
        mask_neg = 1.0 - mask_pos
        
        # Remove self-similarity
        mask_pos.fill_diagonal_(0)
        
        total_loss = 0.0
        num_anchors = 0
        
        for i in range(batch_size):
            # Find positives and negatives for anchor i
            pos_mask = mask_pos[i]
            neg_mask = mask_neg[i]
            
            num_pos = pos_mask.sum()
            if num_pos == 0:
                continue  # Skip if no positives
            
            # Compute log probability
            pos_sim = sim_matrix[i] * pos_mask
            neg_sim = sim_matrix[i] * neg_mask
            
            # LogSumExp for numerical stability
            all_sim = torch.cat([pos_sim[pos_mask.bool()], neg_sim[neg_mask.bool()]])
            log_prob = pos_sim[pos_mask.bool()] - torch.logsumexp(all_sim, dim=0)
            
            # Average over positives
            loss_i = -log_prob.mean()
            total_loss += loss_i
            num_anchors += 1
        
        return total_loss / max(num_anchors, 1)


class TokenContrastiveLoss(nn.Module):
    """Token-subset contrastive loss (TokCon)"""
    
    def __init__(self, tau: float = 0.07):
        super().__init__()
        self.tau = tau
    
    def forward(self, odd_tokens: torch.Tensor, pos_tokens: torch.Tensor, 
                neg_tokens_list: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            odd_tokens: [embed_dim] averaged odd token embedding
            pos_tokens: [embed_dim] averaged positive token embedding  
            neg_tokens_list: List of [embed_dim] negative token embeddings
        Returns:
            loss: scalar loss value
        """
        # Positive similarity
        pos_sim = torch.dot(odd_tokens, pos_tokens) / self.tau
        
        # Negative similarities
        neg_sims = []
        for neg_tokens in neg_tokens_list:
            neg_sim = torch.dot(odd_tokens, neg_tokens) / self.tau
            neg_sims.append(neg_sim)
        
        if neg_sims:
            neg_sims = torch.stack(neg_sims)
            # InfoNCE loss
            all_sims = torch.cat([pos_sim.unsqueeze(0), neg_sims])
            log_prob = pos_sim - torch.logsumexp(all_sims, dim=0)
            return -log_prob
        else:
            # Fallback if no negatives
            return torch.tensor(0.0, device=odd_tokens.device, requires_grad=True)


class WallyClLoss(nn.Module):
    """Combined WallyCL loss function"""
    
    def __init__(self, 
                 lambda_sup: float = 1.0,
                 lambda_tok: float = 0.5,
                 lambda_ce: float = 1.0,
                 tau_sup: float = 0.1,
                 tau_tok: float = 0.07,
                 alpha: float = 5.0,
                 distance_type: str = 'cosine'):
        super().__init__()
        
        self.lambda_sup = lambda_sup
        self.lambda_tok = lambda_tok
        self.lambda_ce = lambda_ce
        
        self.odd_loss = OddOneOutLoss(alpha=alpha, distance_type=distance_type)
        self.supcon_loss = GroupSupConLoss(tau=tau_sup)
        self.tokcon_loss = TokenContrastiveLoss(tau=tau_tok)
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(self, 
                group_embeddings: List[torch.Tensor],
                group_labels: List[torch.Tensor],
                cls_logits: torch.Tensor,
                cls_labels: torch.Tensor,
                token_contrastive_pairs: Optional[List[Tuple]] = None) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            group_embeddings: List of [K, embed_dim] tensors for each group
            group_labels: List of [K] label tensors for each group
            cls_logits: [total_samples, num_classes] classification logits
            cls_labels: [total_samples] classification labels
            token_contrastive_pairs: Optional list of (odd_tokens, pos_tokens, neg_tokens_list)
        Returns:
            total_loss: scalar total loss
            loss_dict: dictionary of individual loss components
        """
        losses = {}
        device = None
        
        # 1. Odd-One-Out Loss
        odd_losses = []
        for group_emb in group_embeddings:
            if device is None:
                device = group_emb.device
            odd_idx = len(group_emb) - 1  # Last index is odd
            odd_loss = self.odd_loss(group_emb, odd_idx)
            odd_losses.append(odd_loss)
        
        losses['odd'] = torch.stack(odd_losses).mean() if odd_losses else torch.tensor(0.0, requires_grad=True, device=device)
        
        # 2. Supervised Contrastive Loss
        if group_embeddings:
            all_embeddings = torch.cat(group_embeddings, dim=0)
            all_labels = torch.cat(group_labels, dim=0)
            losses['supcon'] = self.supcon_loss(all_embeddings, all_labels)
        else:
            losses['supcon'] = torch.tensor(0.0, requires_grad=True, device=device)
        
        # 3. Token Contrastive Loss
        if token_contrastive_pairs:
            tok_losses = []
            for odd_tok, pos_tok, neg_tok_list in token_contrastive_pairs:
                tok_loss = self.tokcon_loss(odd_tok, pos_tok, neg_tok_list)
                tok_losses.append(tok_loss)
            losses['tokcon'] = torch.stack(tok_losses).mean() if tok_losses else torch.tensor(0.0, requires_grad=True, device=device)
        else:
            losses['tokcon'] = torch.tensor(0.0, requires_grad=True, device=device)
        
        # 4. Classification Loss
        losses['ce'] = self.ce_loss(cls_logits, cls_labels)
        
        # Total loss
        total_loss = (losses['odd'] + 
                     self.lambda_sup * losses['supcon'] + 
                     self.lambda_tok * losses['tokcon'] + 
                     self.lambda_ce * losses['ce'])
        
        # Move losses to same device as total_loss for logging
        device = total_loss.device
        for key in losses:
            if hasattr(losses[key], 'to'):
                losses[key] = losses[key].to(device)
        
        loss_dict = {k: v.item() if hasattr(v, 'item') else v for k, v in losses.items()}
        loss_dict['total'] = total_loss.item()
        
        return total_loss, loss_dict
