from multiprocessing import Lock
import numpy as np
import queue
from typing import List

from yarr.agents.agent import ScalarSummary
from yarr.agents.agent import Summary
from yarr.utils.stat_accumulator import StatAccumulator
from yarr.utils.transition import ReplayTransition


class Metric:
    def __init__(self, maxlen=None):
        if maxlen is None:
            self._previous = []
        else:
            self._previous = queue.deque(maxlen=maxlen)
        self._current = 0

    def update(self, value):
        self._current += value

    def next(self):
        self._previous.append(self._current)
        self._current = 0

    def reset(self):
        if isinstance(self._previous, list):
            self._previous.clear()

    def min(self):
        return np.min(self._previous)

    def max(self):
        return np.max(self._previous)

    def mean(self):
        return np.mean(self._previous)

    def median(self):
        return np.median(self._previous)

    def std(self):
        return np.std(self._previous)

    def __len__(self):
        return len(self._previous)

    def __getitem__(self, i):
        return self._previous[i]


class _SimpleAccumulator(StatAccumulator):
    def __init__(self, prefix, maxlen=None):
        self._prefix = prefix
        self._lock = Lock()
        self._transitions = 0

        self._metrics = {
            "return": Metric(maxlen=maxlen),
            "length": Metric(maxlen=maxlen),
            "translation": Metric(maxlen=maxlen),
            "max_velocity": Metric(maxlen=maxlen),
        }

    def _reset_data(self):
        with self._lock:
            for metric in self._metrics.values():
                metric.reset()

    def step(self, transition: ReplayTransition, eval: bool):
        if transition.timeout:
            # timeout by invalid actions
            self._metrics["return"].reset()
            self._metrics["length"].reset()
            self._metrics["translation"].reset()
            self._metrics["max_velocity"].reset()
            return

        if transition.info.get("is_invalid", False):
            # invalid action
            return

        with self._lock:
            self._transitions += 1
            self._metrics["return"].update(transition.reward)
            self._metrics["length"].update(1)
            if "translation" in transition.info:
                self._metrics["translation"].update(
                    transition.info["translation"]
                )
            if "max_velocity" in transition.info:
                self._metrics["max_velocity"].update(
                    transition.info["max_velocity"]
                )
            if transition.terminal:
                self._metrics["return"].next()
                self._metrics["length"].next()
                self._metrics["translation"].next()
                self._metrics["max_velocity"].next()

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
    def __init__(self, maxlen=None):
        self._accumulator_train = _SimpleAccumulator(
            prefix="train_envs", maxlen=maxlen
        )
        self._accumulator_eval = _SimpleAccumulator(
            prefix="eval_envs", maxlen=maxlen
        )

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
