from mteb.abstasks.TaskMetadata import TaskMetadata
from mteb.abstasks.AbsTaskClassification import AbsTaskClassification
from typing_extensions import override


class AgNews(AbsTaskClassification):
    metadata = TaskMetadata(
        name="AgNews",
        description="AG is a collection of more than 1 million news articles. ",
        dataset={
            "path": "fancyzhx/ag_news",
            "revision": "eb185aade064a813bc0b7f42de02595523103ca4"
        },
        type="Classification",
        category="s2s",
        modalities=["text"],  # text or image
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=None,
        domains=None,
        task_subtypes=None,
        license=None,
        annotations_creators=None,
        dialect=None,
        sample_creation=None,
        bibtex_citation="""""",
    )
    # option
    samples_per_label: int = 16


class AgNewsLocal(AbsTaskClassification):
    metadata = TaskMetadata(
        name="PersonGender",
        description="Gender classification by name",
        dataset={
            "path": "csv",
            "data_files": {
                "train": "person_train.csv",
                "validation": "person_test.csv",
            },
            "revision": "None",
        },
        type="Classification",
        category="s2s",
        modalities=["text"],  # text or image
        eval_splits=["validation"],
        eval_langs=["cmn-Hans"],
        main_score="accuracy",
        date=None,
        domains=None,
        task_subtypes=None,
        license=None,
        annotations_creators=None,
        dialect=None,
        sample_creation=None,
        bibtex_citation="""""",
    )
    # option
    samples_per_label: int = 16

    @override
    def dataset_transform(self):
        self.dataset = self.dataset.rename_column("name", "text")
        self.dataset = self.dataset.rename_column("gender", "label")


if __name__ == "__main__":
    import mteb

    model = mteb.get_model("BAAI/bge-m3")
    evaluation = mteb.MTEB(
        tasks=[
            AgNews(method="kNN-pytorch", n_experiments=8),
            # AgNewsLocal(method="kNN-pytorch", n_experiments=8),
            AgNewsLocal(n_experiments=8),
        ]
    )
    evaluation.run(model)
