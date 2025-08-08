## 3) Pseudocode for Key Algorithms

### 3.1 Group Construction

```python
# Build K=m+1 sized groups: m positives + 1 odd
function BUILD_GROUPS(batch_indices, labels, m):
    groups = []
    for cls in unique(labels[batch_indices]):
        pos = sample_m_examples_of_class(cls, m)
        odd_cls = sample_class_neq(cls)
        odd = sample_1_example_of_class(odd_cls)
        groups.append(pos + [odd])  # length K
    return groups  # list of lists of indices
```

### 3.2 Differentiable Token Subset Mining (Gumbel-Top-k)

```python
# logits_t: token saliency scores per token t (from a small scorer head)
# returns soft selection weights w_t that approximate top-k one-hot picks
function GUMBEL_TOPK_WEIGHTS(logits_t, k, tau):
    g_t = -log(-log(U_t))  # U_t ~ Uniform(0,1) per token (Gumbel noise)
    y_t = (logits_t + g_t) / tau
    w = softmax(y_t)  # continuous relaxation
    w = SPARSE_TOPK_PROJECTION(w, k)  # optional Sinkhorn/sparsemax step
    return w  # sum(w) ≈ k, each w_t ∈ [0,1]

function TOKEN_AVG(z_tokens, w):
    return sum_t (w_t * z_tokens[t]) / (sum_t w_t + eps)
```

### 3.3 Losses

```python
# All embeddings are L2-normalized (unit vectors)
function ODD_LOSS(group_embeddings):
    # group_embeddings: [K, d]
    m = K - 1
    # average distance to the m positives (assume last is odd label for supervision)
    scores = []
    for k in range(K):
        s = 0
        for i in range(K-1):  # positives are 0..m-1
            s += distance(group_embeddings[k], group_embeddings[i])
        s = s / m
        scores.append(s)
    # softmax over scores, label = index_of_odd
    return cross_entropy(softmax(scores), label=odd_index)

function SUPCON_LOSS(embeddings, labels, tau):
    loss = 0
    for i in range(len(embeddings)):
        pos_idx = [j for j in range(len(embeddings)) if labels[j]==labels[i] and j!=i]
        neg_idx = [j for j in range(len(embeddings)) if labels[j]!=labels[i]]
        num = sum(exp(dot(embeddings[i], embeddings[j]) / tau) for j in pos_idx)
        den = num + sum(exp(dot(embeddings[i], embeddings[j]) / tau) for j in neg_idx)
        loss += -log(num / den)
    return loss / len(embeddings)

function TOKCON_LOSS(odd_tokens, pos_tokens, neg_tokens, k, tau):
    w_odd = GUMBEL_TOPK_WEIGHTS(score(odd_tokens), k, tau)
    w_pos = GUMBEL_TOPK_WEIGHTS(score(pos_tokens), k, tau)
    z_odd = TOKEN_AVG(odd_tokens, w_odd)
    z_pos = TOKEN_AVG(pos_tokens, w_pos)
    num = exp(dot(z_odd, z_pos) / tau)
    den = num + sum(exp(dot(z_odd, TOKEN_AVG(n, GUMBEL_TOPK_WEIGHTS(score(n), k, tau))) / tau) for n in neg_tokens)
    return -log(num / den)
```

### 3.4 Training Loop

```python
for epoch in range(E):
    for images, labels in loader:
        groups = BUILD_GROUPS(batch_indices=range(len(images)), labels=labels, m=m)
        group_embeds, cls_logits, token_sets = [], [], []
        for G in groups:
            imgs = [augment(images[i]) for i in G]
            tokens = [backbone_tokens(img) for img in imgs]
            embeds = [project(pool(t)) for t in tokens]  # L2 norm
            group_embeds.append(embeds)
            cls_logits.extend([classifier(pool(t)) for t in tokens])
            token_sets.append(tokens)

        L_odd = sum(ODD_LOSS(e) for e in group_embeds) / len(group_embeds)
        L_sup = SUPCON_LOSS(flatten(group_embeds), flatten(group_labels(groups)), tau_sup)
        L_tok = 0
        for tokens in token_sets:
            odd_tokens = tokens[-1]
            pos_tokens = MERGE_POSITIVE_TOKENS(tokens[0:m])  # average or attention pool
            neg_tokens = SAMPLE_NEGATIVE_TOKEN_SETS(batch_excluding_current_group)
            L_tok += TOKCON_LOSS(odd_tokens, pos_tokens, neg_tokens, k, tau_tok)
        L_tok /= len(token_sets)

        L_ce = cross_entropy(concat(cls_logits), concat(labels_from_groups(groups)))
        loss = L_odd + lam_sup*L_sup + lam_tok*L_tok + lam_ce*L_ce
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        if use_ema: ema.update(model)
```

### 3.5 Validation Loop (Classification + Odd-ID)

```python
with eval_mode(model):
    metrics = reset()
    for images, labels in val_loader:
        logits = classifier(pool(backbone_tokens(images)))
        metrics.update_classification(logits, labels)
        # Odd-ID groups
        groups = BUILD_GROUPS_FOR_VAL(labels, m)
        for G in groups:
            embeds = [project(pool(backbone_tokens(images[i]))) for i in G]
            scores = compute_inconsistency_scores(embeds, m)
            pred_odd = argmax(scores)
            metrics.update_odd_id(pred_odd == true_odd(G))
    report(metrics)
```

---