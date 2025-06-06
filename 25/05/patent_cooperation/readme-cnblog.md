产业集群间的专利合作关系

准备条件：

- 全国的专利表
- 目标集群间的企业名单

根据专利的共同申请人，判断这两家企业之间存在专利合作关系。

利用`1_filter_patent.py`，从全国的3000多万条专利信息中，筛选出与目标集群企业相关的专利。
只要专利的申请人包含目标企业则筛选出该企业。

`2_专利协同.ipynb` 实现统计出两家企业之间的合作次数。效果如下图所示：

![image-20250527182622738](https://img2023.cnblogs.com/blog/2589035/202505/2589035-20250527185949676-1867433411.png)

针对某件专利的多位申请人，使用 split 拆分出每一个申请人。对申请人排序，避免后续计数的时候，对 A->B 与 B->A 都进行计数。只要对申请人进行排序，再使用`itertools.combinations(ps, 2)` 那么就只会出现 A->B 而不会出现 B->A 的情况。

`3_fetch_lat_long.py`: 通过百度地图API接口，获取企业单位对应的经纬度。注册之后，免费key每天调用5k次地址解析，每秒限速3次。使用了**asyncio**异步加速，通过tqdm进度条显示进度。

`.env`的文件内容如下:

```
api_key=W6xxxxxx...xxxxxxRP
```
程序捕获到异常，自动把当前获取的响应数据，保存到本地json文件中。支持手动 Ctrl + C 停止运行，然后保存已获取的数据。
```json
{
  "吴江市xx电子科技有限公司": {
    "status": 0,
    "result": {
      "location": {
        "lng": 120.69799250160581,
        "lat": 31.17929972132341
      },
      "precise": 0,
      "confidence": 50,
      "comprehension": 100,
      "level": "NoClass"
    },
    "idx": 5171,
    "address_input": "吴江市xx电子科技有限公司"
  },
  ...
  }
```

`4_判断企业类型.ipynb`: 增加 `both_in`属性，若两家公司都属于集群企业为1，否则为0。

![image-20250527185013761](https://img2023.cnblogs.com/blog/2589035/202505/2589035-20250527185948984-245388645.png)

`5_专利协同数据添加经纬度_by_baidu.ipynb`: 

为两家公司都添加上经纬度。

![image-20250527185558655](https://img2023.cnblogs.com/blog/2589035/202505/2589035-20250527185948248-93971624.png)

## 代码开源

[https://github.com/JieShenAI/csdn/tree/main/25/05/patent_cooperation](https://github.com/JieShenAI/csdn/tree/main/25/05/patent_cooperation)

