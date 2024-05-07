from typing import List

def score_fn(gold: List[str], pred: List[str], beta: float) -> float:
    """
    Вычисляет F score

    parameters:
        gold (List[str]): Список истинных именованных сущностей
        pred (List[str]): Список задетектированных именованных сущностей
        beta (float): Коэффициент beta, при 0 < beta < 1 приоритет на точность, beta > 1 приоритет на полноту

    returns:
        float: F мера
    """
    if beta <= 0:
        raise ValueError("beta должна быть > 0")

    if len(pred) == 0 or len(gold) == 0:
        return 0.0

    true_positive = 0.0
    for entity_name in pred:
        if entity_name in gold:
            true_positive += 1.

    if true_positive == 0.0:
        return 0.0

    precision = true_positive / len(pred)
    recall = true_positive / len(gold)
    f_score = ((1+beta**2)*precision*recall) / (beta**2*precision + recall)
    return f_score


def error_by_category(gold: List[str], pred: List[str]) -> int:
    """
    Считает процент ошибок в категории.

    parameters:
        gold (List[str]): Список истинных именованных сущностей
        pred (List[str]): Список задетектированных именованных сущностей

    returns:
        int: Процент ошибок. (Если 1, то в данной категории нет предсказанных сущностей,
                              если > 1, то модель нашла больше сущностей данной категории, чем
                                            есть на самом деле и при этом неверных,
                              если << 1, модель нашла сущности данной категории, при этом не больше чем их на самом деле и верных)
    """
    if len(pred) == 0 and len(gold) == 0:
        return 0.
    elif len(pred) == 0:
        return 1.
    elif len(gold) == 0:
        return 0.

    true_positive = 0
    for entity_name in pred:
        if entity_name in gold:
            true_positive += 1

    return (len(pred) - true_positive) / len(gold)