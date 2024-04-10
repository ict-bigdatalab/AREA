# AREA

This is the source code for paper "[Black-box Adversarial Attacks against Dense Retrieval Models: A Multi-view Contrastive Learning Method](https://doi.org/10.1145/3583780.3614793)".
This project is in the process of being organized and will be updated continuously, so stay tuned.
If you have any questions related to the paper, feel free to contact Yu-an Liu (liuyuan1b@ict.ac.cn).

## Introduction

Neural ranking models (NRMs) and dense retrieval (DR) models have given rise to substantial improvements in overall retrieval performance. In addition to their effectiveness, and motivated by the proven lack of robustness of deep learning-based approaches in other areas, there is growing interest in the robustness of deep learning-based approaches to the core retrieval problem. Adversarial attack methods that have so far been developed mainly focus on attacking NRMs, with very little attention being paid to the robustness of DR models.


In this paper, we introduce the adversarial retrieval attack (AREA) task. The AREA task is meant to trick DR models into retrieving a target document that is outside the initial set of candidate documents retrieved by the DR model in response to a query. We consider the decision-based black-box adversarial setting, which is realistic in real-world search engines. To address the AREA task, we first employ existing adversarial attack methods designed for NRMs. We find that the promising results that have previously been reported on attacking NRMs, do not generalize to DR models: these methods underperform a simple term spamming method. We attribute the observed lack of generalizability to the interaction-focused architecture of NRMs, which emphasizes fine-grained relevance matching. DR models follow a different representation-focused architecture that prioritizes coarse-grained representations. We propose to formalize attacks on DR models as a contrastive learning problem in a multi-view representation space. The core idea is to encourage the consistency between each view representation of the target document and its corresponding viewer via view-wise supervision signals. Experimental results demonstrate that the proposed method can significantly outperform existing attack strategies in misleading the DR model with small indiscernible text perturbations. 

## License

This project is under Apache License 2.0.

## Citation

If you find our work useful, please consider citing our paper:
```
@inproceedings{10.1145/3583780.3614793,
  author = {Liu, Yu-An and Zhang, Ruqing and Guo, Jiafeng and de Rijke, Maarten and Chen, Wei and Fan, Yixing and Cheng, Xueqi},
  title = {Black-Box Adversarial Attacks against Dense Retrieval Models: A Multi-View Contrastive Learning Method},
  year = {2023},
  isbn = {9798400701245},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  url = {https://doi.org/10.1145/3583780.3614793},
  doi = {10.1145/3583780.3614793},
  pages = {1647–1656},
  numpages = {10},
  series = {CIKM '23}
}
```
