from typing import TypeVar
from concurrent.futures import ProcessPoolExecutor, as_completed

from tqdm import tqdm

T = TypeVar("T")


def calculate_ranks_sample(trues: list[T], preds: list[T], k: int) -> float:
    for i, pred in enumerate(preds):
        if i == k:
            return 0.
        if pred in trues:
            return 1 / (i + 1)
    return 0.


def mrr(y_true: list[list[T]], y_pred: list[list[T]], k: int = 100, debug: bool = False) -> float:
    ranks = []
    with ProcessPoolExecutor() as executor:
        futures = []
        for trues, preds in zip(y_true, y_pred):
            futures.append(executor.submit(calculate_ranks_sample, trues, preds, k))

        ranks = []
        for future in tqdm(as_completed(futures), total=len(y_pred)):
            ranks.append(future.result())
    if debug:
        print(ranks)
    return sum(ranks) / len(ranks)


if __name__ == "__main__":

    # https://ru.wikipedia.org/wiki/%D0%A1%D1%80%D0%B5%D0%B4%D0%BD%D0%B5%D0%BE%D0%B1%D1%80%D0%B0%D1%82%D0%BD%D1%8B%D0%B9_%D1%80%D0%B0%D0%BD%D0%B3
    outp = mrr(
        y_true=[
            ["кочерёг"],
            ["попадей"],
            ["турок"]
        ],
        y_pred=[
            ["кочерг", "кочергей", "кочерёг"],
            ["попадь", "попадей", "попадьёв"],
            ["турок", "турков", "турчан"]
        ]
    )

    print(outp)  # должно быть 0.61
