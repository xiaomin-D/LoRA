## 说明
代码仓库基于lora仓库建立  
模型基于gpt2-M  

### 训练lora流程

```bash
# 安装当前版本lora
pip install -e .
# 安装依赖
cd examples/NLG
pip install -r requirement.txt
bash download_pretrained_checkpoints.sh
bash create_datasets.sh
cd ./eval
bash download_evalscript.sh
cd ..
bash train.sh
```
评测等在[readme中](examples/NLG/README.md)

