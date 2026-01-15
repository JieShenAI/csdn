import sys

sys.path.append("../src/")

from classify.data import DiyDataCollator
from classify.model import DiyModel
from classify.trainer import DiyTrainerUtil

trainer_util = DiyTrainerUtil(
    model_class=None,
    dataset_class=None,
    datacollator_class=DiyDataCollator,
)


def train():
    trainer_util.train()


def eval(dataset):
    trainer_util.evaluate(dataset)
