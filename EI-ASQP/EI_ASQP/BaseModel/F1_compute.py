from sklearn.metrics import precision_score, recall_score, f1_score

def is_subsequence(target_words, pred_words):
    """
    判断 target_words 是否为 pred_words 的子序列
    """
    if not target_words:
        return False
    p_idx = 0
    for w in target_words:
        while p_idx < len(pred_words) and pred_words[p_idx] != w:
            p_idx += 1
        if p_idx == len(pred_words):
            return False
        p_idx += 1
    return True

#如果 targets 的所有词都按顺序出现在 predictions 中则预测正确
def compute_prf1(predictions, targets):
    # 如果传入的是单个字符串，转换为列表
    if isinstance(predictions, str):
        predictions = [predictions]
    if isinstance(targets, str):
        targets = [targets]

    y_true = []
    y_pred = []

    for pred, target in zip(predictions, targets):
        pred = pred.strip().lower()
        target = target.strip().lower()

        y_true.append(1 if target else 0)

        pred_words = pred.split()
        target_words = target.split()

        match = is_subsequence(target_words, pred_words)
        y_pred.append(1 if match else 0)

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    return precision, recall, f1