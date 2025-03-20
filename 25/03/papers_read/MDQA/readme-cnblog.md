# MDQA 知识图谱提示用于多文档问答

论文名：Knowledge Graph Prompting for Multi-Document Question Answering

论文地址：https://arxiv.org/abs/2308.11730

源码：https://github.com/YuWVandy/KG-LLM-MDQA

## 论文阅读

该论文提出了一种知识图谱提示（KGP）方法，以构建正确的上下文来提示LLMs进行MD-QA，该方法包括一个图构建模块和一个图遍历模块。在图构建方面，创建了一个跨越多文档的知识图谱（KG），边表示段落或文档结构之间的语义/词汇相似性。在图遍历方面，我们设计了一个基于LLMs的图遍历代理，该代理在节点间导航并收集支持性段落，以帮助LLMs进行MD-QA。所构建的图作为全局规则器，调节段落之间的过渡空间并减少检索延迟。同时，图遍历代理充当一个本地导航器，收集相关上下文以逐步接近问题并保证检索质量。



我们平常做RAG文本召回的时候，也不会只针对一个文档做召回，本质上也是多文档的召回。该文章在传统的RAG召回的基础之上，增加了文章、段落节点。在每个段落之间添加了边，从而实现一种递归的文本召回（找到一个与问题相似的段落节点后，在该段落节点的邻接的节点，也进行相似查找）。如下图右侧所示，一篇文章上面所有内容，包括表格、段落等都挂在到一个文章节点上。（以前我也有过这样的想法，也做了文章结构的知识图谱，但没有找到可以讲故事的地方）。下图右侧的段落节点之间的边，代表这两个节点很相似。

 段落之间用相似度构建边，做成可视化，呈现给用户一种直观的感觉是可以的。但是他们把这种加入到召回文本中，让大模型去回答，我个人认为这里不一定能够提升效果。因为他们对文本召回的检索器进行了微调，所以模型的效果肯定好，他们应该要做一个段落临接节点的消融实验，证明在段落节点之间添加相似边是有效的。

![image-20250319200919588](https://img2023.cnblogs.com/blog/2589035/202503/2589035-20250320103610575-1649799605.png)



实验部分：

![image-20250319203045818](https://img2023.cnblogs.com/blog/2589035/202503/2589035-20250320103610181-387669124.png)

在这篇文章的源码中，可以学到数据集的构建，KNN、TF-IDF、BM25等这些检索器的使用。

该论文没有给出召回率方面的评估结果，直接给出最终的结果。他们评估大模型回答问题答案的效果，采用的是大模型打分的方法，提示词如下：

```python
def prompt_eval():
    eval_prompt = """You are an expert professor specialized in grading whether the prediction to the question is correct or not according to the real answer.
    ==================
    For example:
    ==================
    Question: What company owns the property of Marvel Comics?
    Answer: The Walt Disney Company
    Prediction: The Walt Disney Company
    Return: 1
    ==================
    Question: Which constituent college of the University of Oxford endows four professorial fellowships for sciences including chemistry and pure mathematics?
    Answer: Magdalen College
    Prediction: Magdalen College.
    Return: 1
    ==================
    Question: Which year was Marvel started?
    Answer: 1939
    Prediction: 1200
    Return: 0
    ==================
    You are grading the following question:
    Question: {question}
    Answer: {answer}
    Prediction: {prediction}
    If the prediction is correct according to answer, return 1. Otherwise, return 0.
    Return: your reply can only be one number '0' or '1'
    """

    return eval_prompt
```

​    If the prediction is correct according to answer, return 1. Otherwise, return 0.

把大模型生成的答案与真实的答案一起提交给评估的模型，如果预测的结果是对的返回1，预测结果不对返回0。

评估结果的测试脚本 `Pipeline/evaluation/eval.ipynb`：

![image-20250319203808911](https://img2023.cnblogs.com/blog/2589035/202503/2589035-20250320103609828-250870161.png)



## 代码解析

### 图谱构建

`Data-Collect/graph_construct.py`

```python
def knn_graph(i_d, k_knn, embs, strategy='cos'):
    idx, d = i_d

    emb = embs[idx]

    # build a knn Graph
    if strategy == 'cos':
        sim = cosine_similarity(emb, emb)

    elif strategy == 'dp':
        sim = np.matmul(emb, emb.transpose(1, 0))

    # topk
    top_idx = np.argsort(-sim, axis=1)[:, 1:k_knn + 1]

    tail_nodes = np.arange(top_idx.shape[0]).repeat(k_knn) # flatten
    head_nodes = top_idx.reshape(-1)
    edges = [(node1, node2) for node1, node2 in zip(tail_nodes, head_nodes)]

    G = nx.DiGraph()
    G.add_edges_from(edges)

    return idx, G
```

上述代码实现了，两个节点根据它俩之间向量相似度构建边。



### 检索器微调

![image-20250319204356647](https://img2023.cnblogs.com/blog/2589035/202503/2589035-20250320103609516-877169450.png)

主要关注 **桥接问题**，因为比较问题不需要关注顺序，先召回哪一个文本都行。针对桥接问题首先需要能够对Q召回S1，然后再对 Q+S1 能够召回S2。相对传统的检索器微调需要增加Q+S1能够学会召回S2的过程。所以这一点，在下述的数据集构造中多了`q1_c1_enc`，在损失值的计算中多了 `loss_fct(scores_2, target_2)`。



数据集：

![image-20250319204051184](https://img2023.cnblogs.com/blog/2589035/202503/2589035-20250320103609091-872662938.png)

* q_enc: 问题的嵌入向量
* q_c1: 问题+第一个文本的嵌入向量
* c1_enc、c2_enc：真实的第一个文本与第二个文本
* n1_enc、n2_enc：从负样本中随机筛选出的两个负样本

损失函数：

```python
def mp_loss(model, batch):
    embs = model(batch)
    loss_fct = CrossEntropyLoss(ignore_index = -1)

    c_embs = torch.cat([embs["c1_emb"], embs["c2_emb"]], dim = 0) # 2B x d
    n_embs = torch.cat([embs["n1_emb"].unsqueeze(1), embs["n2_emb"].unsqueeze(1)], dim = 1) # B*2*M*h

    scores_1 = torch.mm(embs["q_emb"], c_embs.t()) # B x 2B
    n_scores_1 = torch.bmm(embs["q_emb"].unsqueeze(1), n_embs.permute(0, 2, 1)).squeeze(1) # B x 2B
    scores_2 = torch.mm(embs["q_c1_emb"], c_embs.t()) # B x 2B
    n_scores_2 = torch.bmm(embs["q_c1_emb"].unsqueeze(1), n_embs.permute(0, 2, 1)).squeeze(1) # B x 2B

    # mask the 1st hop
    bsize = embs["q_emb"].size(0)
    scores_1_mask = torch.cat([torch.zeros(bsize, bsize), torch.eye(bsize)], dim=1).to(embs["q_emb"].device)
    scores_1 = scores_1.float().masked_fill(scores_1_mask.bool(), float('-inf')).type_as(scores_1)
    scores_1 = torch.cat([scores_1, n_scores_1], dim=1)
    scores_2 = torch.cat([scores_2, n_scores_2], dim=1)

    target_1 = torch.arange(embs["q_emb"].size(0)).to(embs["q_emb"].device)
    target_2 = torch.arange(embs["q_emb"].size(0)).to(embs["q_emb"].device) + embs["q_emb"].size(0)

    loss = loss_fct(scores_1, target_1) + loss_fct(scores_2, target_2)

    return loss
```



* loss_fct(scores_1, target_1)： 模型学会通过 Q 召回S1；

* loss_fct(scores_2, target_2)：模型学会通过 Q+S1 能够召回S2；

上述的损失函数写的挺复杂的，如果第一次看到这种检索器的损失函数，应该会有很多同学看不懂。

关于检索器微调损失值：这里的损失函数是 CrossEntropyLoss 与分类挺像的，把问题的向量与相关文本做乘法，得到的是问题的向量与相关文本的相似度的值。两个向量做乘法得到的是这两个向量相似度。 这个损失函数的就是让正确文本对应的相似度的值足够大，损失值才会小。



如果BGE检索器的微调还不会的话，也不用硬看上述代码，时间充裕的话，可以先看懂BGE检索器微调。[transformers二次开发——（定义自己的数据加载器 模型 训练器）bge模型微调流程](https://www.bilibili.com/video/BV1eu4y1x7ix) 这是一个B站的视频讲解的BGE微调的，但是该视频有一点遗憾的地方，在关键的损失值计算部分，该UP主讲解错，后来他也在评论区进行了回应。如果大家想深入了解BGE微调，进入 https://github.com/FlagOpen/FlagEmbedding 仓库，找到23年10月的版本（新版本代码太多了，旧版本代码很简洁），一步一步debug，后面自然就会懂。



为了防止我以后忘记，简单写几句：

`scores_1 = torch.mm(embs["q_emb"], c_embs.t())`  把问题的向量与所有候选文本的向量做一个乘法。

`scores_1_mask = torch.cat([torch.zeros(bsize, bsize), torch.eye(bsize)], dim=1).to(embs["q_emb"].device)` 这里使用了mask，把`c2_emd` 给遮罩掉。（在看懂代码前，我就想到了要遮罩c2_emb，然后发现他果然做了遮罩）

因为通过 q_emb 学会召回 c1_emb。通过 q_c1_emb 才应该学会召回c2_emb。

对于scores_1的损失函数而言，正确的 label 给了c1_emb，c2_emb自然就是错误。c2_emb会成为负样本，这是不允许的，这样会把 q_emb 与 c2_emb 的相似程度给拉远了，这样不行，最好的做法还是把 c2_emb 给遮罩掉。



对于 target_2 `torch.arange(embs["q_emb"].size(0)).to(embs["q_emb"].device) + embs["q_emb"].size(0)` 在label数值加的embs["q_emb"].size(0)是batch_size。

`score_1`的shape是 (batch_size, 2 x batch_size) 针对最后一个维度有2 x batch_size而言，前面一个batch_size是score_1，后面一个batch_size是score_2，所有target_2 的值相比 target_1 要再加 batch_size。



### 检索器使用

`KG-LLM-MDQA\Pipeline\retriever.py` 大家可以看一下这个脚本中，**在做向量召回的时候，使用的召回方法绝大多数都是TF-IDF**，那这个言外之意就是前面检索器的微调效果不好。那岂不是前面微调了半天的检索器，白微调了。论文的实验结果中，效果比较好的KGP_T5方法使用的检索器 `llm_retriever_KG_T5` 也是用的 TF-IDF。

```python
class KG_retriever(object):
    def __init__(self, k):
        self.k = k
    
    def retrieve(self, data, G):
        corpus = [c for _, c in data['title_chunks']]
        candidates_idx = list(range(len(corpus)))

        seed = data['question']
        retrieve_idxs = []

        prev_length = 0
        count = 0
        retrieve_num = [10, 5, 5, 5, 3, 2, 2, 2, 2, 2, 2]
        while len(retrieve_idxs) < self.k:
            idxs = tf_idf(seed, candidates_idx, corpus, k = retrieve_num[count], visited = retrieve_idxs)
            retrieve_idxs.extend(idxs[:max(0, self.k - len(retrieve_idxs))])
            
            candidates_idx = set(chain(*[list(G.neighbors(node)) for node in idxs]))
            candidates_idx = list(candidates_idx.difference(retrieve_idxs))

            if len(retrieve_idxs) == prev_length:
                break
            else:
                prev_length = len(retrieve_idxs)
            
            count += 1

        return [corpus[idx] for idx in retrieve_idxs], None, None, None
```

candidates_idx 候选的节点，利用 tf_idf 算法从候选节点中，找出新的候选节点。visited 表示已经访问过的节点，已经访问过的节点不再加入到新的候选节点中。如果新的候选节点为空，则停止节点召回。类似广度优先搜索，一层一层的往下搜索。retrieve_num 表示每一层要筛选的节点数量，第一层多取一点，下面的几层少选一点。

### 大模型检索微调



![image-20250319212025997](https://img2023.cnblogs.com/blog/2589035/202503/2589035-20250320103608725-135356610.png)



![image-20250319212153865](https://img2023.cnblogs.com/blog/2589035/202503/2589035-20250320103608123-1019067943.png)

我没有阅读他们的代码，但通过阅读上述的提示词，我认为他们在微调大模型让其学会根据问题生成相关支撑文本，再用生成的支撑文本做文本检索召回。

