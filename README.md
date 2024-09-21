# Dynamic Fusion of Shared and Modality-Specific Features for Multi-Modal Vulnerability Detection
## Code repository for the study
In this study, we empirically study different existing Deep Learning Based Vulnerability Detection techniques for real world vulnerabilities. We test the feasibility of existing techniques in three different datasets.
- CodeXGLUE[1]:[https://drive.google.com/file/d/1o3vGR5WINHGIRKDlnapuYXcQ3-fPjlnD/view?usp=drive_link](https://drive.google.com/file/d/1o3vGR5WINHGIRKDlnapuYXcQ3-fPjlnD/view?usp=drive_link)
- Reveal[2]:[https://drive.google.com/drive/folders/1fQBKTj9RHOPORjgzgH5UpD5hXzZh1N7e?usp=drive_link](https://drive.google.com/drive/folders/1fQBKTj9RHOPORjgzgH5UpD5hXzZh1N7e?usp=drive_link)
- Fan et al.[3]:[https://drive.google.com/file/d/18tIzDtwfikVvSAnBJaJmloe-SiwKuSag/view?usp=drive_link](https://drive.google.com/file/d/18tIzDtwfikVvSAnBJaJmloe-SiwKuSag/view?usp=drive_link)
## Requirement
Our code is based on Python3 (>= 3.7). There are a few dependencies to run the code. The major libraries are listed as follows:
- torch (==1.9.0)
- dgl (==0.7.2)
- numpy (==1.22.3)
- sklearn (==0.0)
- pandas (==1.4.1)
- tqdm
- transformers
## Preprocessing
To generate the AST sequence, please run [ast_sequence.py](https://github.com/1code12/DFSM-MVD/blob/main/Data_process/ast_sequence.py).
For processing code and generating the Code Property Graph (CPG), please refer to the [AMPLE](https://github.com/AMPLE001/AMPLE) or [Reveal](https://github.com/VulDetProject/ReVeal).
## Acknoledgements
We are using several different components from the state-of-the-art research. Please cite accordingly to pay due attributes and credits to the authors.

If you use Reveal, please cite the following
```@article{chakraborty2020deep, title={Deep Learning based Vulnerability Detection: Are We There Yet?},
author={Chakraborty, Saikat and Krishna, Rahul and Ding, Yangruibo and Ray, Baishakhi},
journal={arXiv preprint arXiv:2009.07235},
year={2020}
}
```
If you use AMPLE, please cite the following
```
@INPROCEEDINGS{AMPLE,
  author={Wen, Xin-Cheng and Chen, Yupan and Gao, Cuiyun and Zhang, Hongyu and Zhang, Jie M. and Liao, Qing},
  booktitle={2023 IEEE/ACM 45th International Conference on Software Engineering (ICSE)}, 
  title={Vulnerability Detection with Graph Simplification and Enhanced Graph Representation Learning}, 
  year={2023},
  pages={2275-2286},
  doi={10.1109/ICSE48619.2023.00191}}
```
## References
[1] Lu S, Guo D, Ren S, et al. Codexglue: A machine learning benchmark dataset for code understanding and generation[J]. arXiv preprint arXiv:2102.04664, 2021.

[2] Saikat Chakraborty, Rahul Krishna, Yangruibo Ding, and Baishakhi Ray. 2020. Deep Learning based Vulnerability Detection: Are We There Yet? arXiv preprint arXiv:2009.07235 (2020).

[3] Jiahao Fan, Yi Li, Shaohua Wang, and Tien Nguyen. 2020. A C/C++ Code Vulnerability Dataset with Code Changes and CVE Summaries. In The 2020 International Conference on Mining Software Repositories (MSR). IEEE.

[4] Yaqin Zhou, Shangqing Liu, Jingkai Siow, Xiaoning Du, and Yang Liu. 2019. Devign: Effective vulnerability identification by learning comprehensive program semantics via graph neural networks. In Advances in Neural Information Processing Systems. 10197–10207.


