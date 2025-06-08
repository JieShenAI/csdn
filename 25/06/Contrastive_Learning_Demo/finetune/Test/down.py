#Model Download
from modelscope import snapshot_download
model_dir = snapshot_download('AI-ModelScope/bert-base-uncased')

print(model_dir)

