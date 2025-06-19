from modelscope import snapshot_download
import mteb

model_dir = snapshot_download("BAAI/bge-m3")
# model_dir = snapshot_download("AI-ModelScope/bert-base-uncased")


model = mteb.get_model(
    model_dir
)  # if the model is not implemented in MTEB it will be eq. to SentenceTransformer(model_name)

tasks = mteb.get_tasks(tasks=["Banking77Classification"])
tasks[0].method = "kNN-pytorch"
evaluation = mteb.MTEB(tasks=tasks)
results = evaluation.run(model, output_folder=f"results/bge-m3")
print(results)
