from multiprocessing import Lock
from typing import List

from yarr.agents.agent import ScalarSummary
from yarr.agents.agent import Summary
from yarr.utils.stat_accumulator import Metric
from yarr.utils.stat_accumulator import StatAccumulator
from yarr.utils.transition import ReplayTransition


class _SimpleAccumulator(StatAccumulator):
    def __init__(self, prefix):
        self._prefix = prefix
        self._lock = Lock()
        self._transitions = 0

        self._metrics = {
            "return": Metric(),
            "length": Metric(),
            "translation": Metric(),
        }

    def _reset_data(self):
        with self._lock:
            self._metrics["return"].reset()
            self._metrics["length"].reset()
            self._metrics["translation"].reset()

    def step(self, transition: ReplayTransition, eval: bool):
        with self._lock:
            self._transitions += 1
            self._metrics["return"].update(transition.reward)
            self._metrics["length"].update(1)
            self._metrics["translation"].update(transition.info["translation"])
            if transition.terminal:
                self._metrics["return"].next()
                self._metrics["length"].next()
                self._metrics["translation"].next()

    def _get(self) -> List[Summary]:
        summaries = []
        for key, metric in self._metrics.items():
            summaries.append(
                ScalarSummary(f"{self._prefix}/{key}", metric.mean())
            )
        summaries.append(
            ScalarSummary(
                f"{self._prefix}/total_transitions", self._transitions
            )
        )
        return summaries

    def pop(self) -> List[Summary]:
        data = []
        if len(self._metrics["length"]) > 1:
            data = self._get()
            self._reset_data()
        return data

    def peak(self) -> List[Summary]:
        return self._get()

    def reset(self):
        self._transitions = 0
        self._reset_data()


class SimpleAccumulator(StatAccumulator):
    def __init__(self):
        self._accumulator_train = _SimpleAccumulator("train_envs")
        self._accumulator_eval = _SimpleAccumulator("eval_envs")

    def step(self, transition: ReplayTransition, eval: bool):
        if not eval:
            self._accumulator_train.step(transition, eval)
        else:
            self._accumulator_eval.step(transition, eval)

    def pop(self) -> List[Summary]:
        return self._accumulator_train.pop() + self._accumulator_eval.pop()

    def peak(self) -> List[Summary]:
        return self._accumulator_train.peak() + self._accumulator_eval.peak()

    def reset(self) -> None:
        self._accumulator_train.reset()
        self._accumulator_eval.reset()
