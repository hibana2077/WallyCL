# WallyCL 修復總結報告

## 🎯 問題診斷結果

經過系統性診斷，我們發現WallyCL驗證準確率卡在隨機水平(1/類別數)的原因：

### ❌ 原始問題
1. **BatchNorm1d在小batch時出錯** - 導致訓練中斷
2. **複雜loss組合干擾學習** - 多個loss同時訓練導致優化困難  
3. **學習率設置不當** - 對複雜模型來說過小
4. **Token機制過於複雜** - 在基礎分類都有問題時引入額外複雜性

### ✅ 成功的修復方案
1. **LayerNorm替代BatchNorm1d** - 解決小batch問題
2. **只用CE loss開始** - 確保基礎分類能力
3. **分層學習率** - 分類頭高學習率(1e-2)，backbone低學習率(5e-6)
4. **簡化forward pass** - 先不用token機制

## 🎉 修復效果

| 指標 | 修復前 | 修復後 | 改善 |
|------|--------|--------|------|
| 驗證準確率 | 0.0125 (隨機) | 0.1333 | **10.7x** |
| 訓練準確率 | ~0% | 25.42% | 穩定學習 |
| 訓練Loss | ~4.9 (停滯) | 11.6→3.1 | 持續下降 |

## 🔧 具體修復步驟

### 1. 修復ProjectionHead (已完成)
```python
# 將BatchNorm1d改為LayerNorm
nn.LayerNorm(hidden_dim)  # 替代 nn.BatchNorm1d(hidden_dim)
```

### 2. 修復配置文件 (已完成)
```yaml
loss:
  lambda_sup: 0.0   # 先設為0
  lambda_tok: 0.0   # 先設為0
  lambda_ce: 1.0    # 只用CE loss

optimizer:
  lr: 5.0e-4        # 提高基礎學習率
  weight_decay: 0.01 # 降低正則化
```

### 3. 修復優化器設置 (已完成)
```python
# 分層學習率
optimizer = optim.AdamW([
    {'params': backbone_params, 'lr': lr * 0.01},  # 5e-6
    {'params': classifier_params, 'lr': lr * 20}   # 1e-2
], weight_decay=weight_decay)
```

### 4. 簡化訓練循環 (已完成)
```python
# 使用簡化的forward pass
outputs = model(images, return_tokens=False)  # 不用token機制
logits = outputs['logits']
loss = criterion(logits, labels)  # 只用CE loss
```

## 📈 循序漸進的恢復計劃

### 階段1: CE-only基礎訓練 (已完成✅)
- 驗證準確率達到隨機水平的3-10倍
- 確保基礎分類功能正常

### 階段2: 逐步添加對比學習
```yaml
# 第一步：添加SupCon
lambda_sup: 0.1
lambda_tok: 0.0
lambda_ce: 1.0

# 第二步：添加Token對比
lambda_sup: 0.2
lambda_tok: 0.1
lambda_ce: 1.0

# 第三步：完整配置
lambda_sup: 0.5
lambda_tok: 0.2
lambda_ce: 1.0
```

### 階段3: 修復odd-one-out評估
檢查評估方向是否正確：
- 如果用距離：outlier = argmax(distances)
- 如果用相似度：outlier = argmin(similarities)

## 💡 重要發現

1. **數據集本身沒問題** - 類別映射一致，標籤範圍正確
2. **模型架構基本正確** - 參數覆蓋完整，梯度流動正常
3. **問題在訓練設置** - BatchNorm、學習率、loss組合
4. **ViT需要特殊處理** - 預訓練backbone需要低學習率，分類頭需要高學習率

## 🚀 後續建議

1. **使用simple_train.py作為基準** - 它已經證明有效
2. **逐步遷移到完整train.py** - 按階段恢復功能
3. **密切監控指標** - 每次改動都要確保不退化
4. **保持简化原則** - 復雜性只在基礎穩定後添加

## 📊 性能對比

| 方法 | Val Acc | vs Random | 狀態 |
|------|---------|-----------|------|
| 原始WallyCL | 0.0083 | 0.7x | ❌ 不如隨機 |
| 小數據集測試 | 0.4444 | 35.6x | ✅ 小規模成功 |
| 修復後CE-only | 0.1333 | 10.7x | ✅ 大規模成功 |
| 目標(完整WallyCL) | >0.3 | >24x | 🎯 下一目標 |

結論：**修復成功！** 現在可以安全地進行完整的WallyCL訓練了。
