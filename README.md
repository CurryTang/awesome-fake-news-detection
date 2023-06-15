# awesome-fake-news-detection

Collection of fake news detection papers

## Awesome Fake News & Rumor Detection

Note: Papers for fact verification will be added later

* Multi-view learning with distinguishable feature fusion for rumor detection [Paper](https://www.sciencedirect.com/science/article/pii/S0950705121011552) Knowledge-Based System 2022 Context-aware fake news detection

 Based on GCAN (ACL 2020) and BiGCN(AAAI 2020), this paper proposes a user-aspect fake news detection model with multi-view features, including the profile view, structural view, and temporal view. A hierarchical fusion module with view-wise attention and capsule attention is adopted to fuse features from each view. 
 
* Domain Adaptive Fake News Detection via Reinforcement Learning [Paper](https://arxiv.org/abs/2202.08159) WWW 2022 domain adaption

  cross-domain fake news detection
   a relatively simple encoder: BERT + attention + lstm
   RL-based domain adaption: Compared to adversarial learning based domain adaption, the RL agent directly transforms the representation from the original domain to domain-invariant representations

* (*) "This is Fake! Shared it by Mistake": Assessing the Intent of Fake News Spreaders [Paper](https://arxiv.org/pdf/2202.04752.pdf) WWW 2022 Fake News Detection, Intent Detection

Combining simple psychological analyses with deep graph learning
   Using influence graph as the encoder to capture users' intents

* (*) Towards Fine-Grained Reasoning for Fake News Detection [Paper](https://arxiv.org/abs/2110.15064) AAAI 2022 

The fake news detection module usually suffers from poor explicability. This paper takes the idea from a related area, fact verification, to develop an explainable fake news detection model. This paper constructs a claim-evidence graph that fuses both textual and topological information in the first step. Then, this paper adopts a varied version of KGAT (ACL 2020) to predict the veracity of each news post. This paper fuses traditional feature engineering with deep learning. 

* MDFEND: Multi-domain Fake News Detection [Paper](https://arxiv.org/abs/2201.00987) [Code](https://github.com/kennqiang/mdfend-weibo21) CIKM 2021 short Domain adaption

This short paper proposes a benchmark for multi-domain fake news detection. It also proposes a simple framework based on the mixture of experts. 

* CED: Credible Early Detection of Social Media Rumors [Paper](http://114.215.64.60:8094/~chm/publications/tkde2019_CED.pdf) [Code](https://github.com/thunlp/CED) IEEE TKDE 2021 Early Rumor Detection

The most exciting part of this paper is the concept of a "credible detection point", which means that the debate on relevant news posts terminates after this point, and the result should be determined. A relevant loss function is proposed according to this concept. 

Because of the long evaluation period of TKDE, the encoder adopted in this paper is out-of-date, so the experimental result is not very appealing. The experimental design, especially the early fake news detection, is a good reference. 

* An Integrated Multi-Task Model for Fake News Detection [Paper](https://ieeexplore.ieee.org/document/9339883) IEEE TKDE 2021 Multi-task Learning

The topic is an essential factor in fake news detection. This paper proposes a semantic graph-based fake news detection method, where each edge's weight equals the semantic similarity between two news posts.
A multi-task learning framework with dynamic task weights is used to train topic classification and fake news detection tasks. 

* Rumor Detection on Twitter with Claim-Guided Hierarchical Graph Attention Networks [Paper](https://aclanthology.org/2021.emnlp-main.786.pdf) EMNLP 2021 rumor detection

An improved version of BiGCN, which adopts a hierarchical attention module to capture the fine-grained local & global pattern among post nodes in the propagation graph. 

* User Preference-aware Fake News Detection [Paper](https://arxiv.org/pdf/2104.12259.pdf) [Code](https://github.com/safe-graph/GNN-FakeNews) SIGIR 2021 short fake news detection

Combining text-based fake news detection and context-based fake news detection
Offering a handy interface compatible with Pytorch-Geometric & DGL, a good starting point for fake news detection-related research

* Towards Propagation Uncertainty: Edge-enhanced Bayesian Graph Convolutional Networks for Rumor Detection ACL 2021 [Paper](https://aclanthology.org/2021.acl-long.297.pdf) [Code](https://github.com/weilingwei96/EBGCN) Bayesian Machine Learning

The main architecture is the same as BiGCN.

The motivation of this paper is to mitigate the unreliable relation within the context of the propagation graph. To solve this issue, two mechanisms are proposed

An edge inference module that updates the adjacency matrix with a convolutional layer

An edge-wise consistency training framework: This part minimizes the KL-divergence between two latent distributions. However, it's not well explained. 

* Compare to The Knowledge: Graph Neural Fake News Detection with External Knowledge [Paper](https://aclanthology.org/2021.acl-long.62.pdf) [Code](https://github.com/BUPT-GAMMA/CompareNet_FakeNewsDetection) ACL 2021 main Knowledge-aware fake news detection

CompareNet is an application of the author's previous work, HGAT. It adopts Tagme(an entity extraction tool) to extract the entities within the context of news content. These entities are ancillary to the detection. 

* InfoSurgeon: Cross-Media Fine-grained Information Consistency Checking for Fake News Detection [Paper](https://aclanthology.org/2021.acl-long.133.pdf) [Code](https://github.com/BUPT-GAMMA/CompareNet_FakeNewsDetection) ACL 2021 main Multi-modal fake news detection, Fake News Generation 

It adopts a multi-modal relation extraction tool to achieve essential entities in the image and relevant image caption. The author finds that inconsistency between image and image caption is a helpful pattern for fake news detection. In the later part of this paper, the author proposes a novel fake news generation method based on a knowledge graph. 

* Mining Dual Emotion for Fake News Detection [Paper](https://www.zhangxueyao.com/assets/www2021-dual-emotion-paper.pdf) [Code](https://github.com/RMSnow/WWW2021) WWW 2021 Fake News Detection, Feature Engineering

Combining feature engineering with deep learning
Constructing a complicated sentiment-based embedding to represent the dual emotion between news posts and comments
It can be used as a plug-in for other fake news detection approaches

* Embracing Domain Differences in Fake News: Cross-domain Fake News Detection using Multi-modal Data [Paper](https://arxiv.org/abs/2102.06314) AAAI 2021 domain adaption, active learning

This paper focuses on a specific problem of fake news detection: domain adaption. The author empirically shows that domain shifting will lead to an obvious performance drop. This paper proposes a framework that consists of three parts: unsupervised domain embedding, domain agnostic news classification, and sample selection for active learning to cope with such problems. It should be noted that one main objective for sample selection is to find those samples which boost domain adaption. 

* Multi-Source Domain Adaptation with Weak Supervision for Early Fake News Detection [Paper](http://web.cs.wpi.edu/~kmlee/pubs/li21bigdata.pdf) [Code](https://github.com/bigheiniu/BigData-MDA-WS) IEEE BigData 2021 domain adaption

Domain adaption + weakly supervised target data

Domain adaption is the core part of this model. It contains two parts

Domain-invariant features: Applying a min-max game(also adopted in "Embracing Domain Differences in Fake News") to achieve domain-invariant features(part A of the loss function)

Domain-specific features: train classification heads for supervised signals for each domain (part B of the loss function)

Weakly supervised target data: Adopting weak labeling function(some rules or hand-crafted features such as the number of second-person pronouns) to generate "weak" labels

The idea of combining deep learning and hand-crafted features is interesting. The experimental part doesn't consider state-of-the-art models, and the result is not good. 

* Temporally evolving graph neural network for fake news detection [Paper](https://www.sciencedirect.com/science/article/abs/pii/S0306457321001965) IPM 2021 Temporal fake news detection

A temporal-propagation-based fake news detection framework that fuses structure, content semantics, and temporal information
The propagation pattern of news is modeled as a **continuous** dynamic news propagation network

For the encoder part, this model adopts a temporal GAT-based model and a temporal difference network to capture the variational information in the series of graphs. 

* Rumour Detection via Zero-shot Cross-lingual Transfer Learning [Paper](https://2021.ecmlpkdd.org/wp-content/uploads/2021/07/sub_661.pdf) ECML-PKDD 2021 Multi-Languages

Zero-shot classification with no labels in the target domain

teacher-student self-training by generating "silver labels"

* Early Detection of Fake News with Multi-source Weak Social Supervision [Paper](https://asu.pure.elsevier.com/en/publications/early-detection-of-fake-news-with-multi-source-weak-social-superv) ECML-PKDD 2020 Weak social supervision

A label weighting network to determine the weight of each soft label

Defining several soft labels using feature engineerings such as sentiment score, credibility score, and bias score

* GCAN: Graph-aware Co-Attention Networks for Explainable Fake News Detection on Social Media [Paper](https://arxiv.org/abs/2004.11648) [Code](https://github.com/l852888/GCAN) ACL 2020 

A cross-modality co-attention framework for content, propagation, and user relationship

Limited explicability from attention modules

* DETERRENT: Knowledge Guided Graph Attention Network for Detecting Healthcare Misinformation [Paper](https://dl.acm.org/doi/pdf/10.1145/3394486.3403092) [Code](https://github.com/cuilimeng/DETERRENT) KDD 2020

Social context-based misinformation detection is not suitable for medical area

Fusing knowledge graph and news articles by article-entity bipartite graph

A heterogeneous graph encoder based on RGCN and GAT

Adopting BPR loss to deal with negative relations

* FANG: Leveraging Social Context for Fake News Detection Using Graph Representation [Paper](https://arxiv.org/abs/2008.07939) [Code](https://github.com/nguyenvanhoang7398/FANG) CIKM 2020(best paper) Fake News Detection

A relatively complete fake news detection including complicated feature engineerings such as stance detection and sentiment analysis

Using GraphSage as the encoder for inductive learning

Considering user, source, and news together

* Category-controlled Encoder-Decoder for Fake News Detection [Paper](https://ieeexplore.ieee.org/document/9511228) TKDE 2020  
category-controlled encoder

sequence encoding

guided matching

fusion merging

category-controlled decoder

pattern-shared unit

decoder

restriction unit

* Weak Supervision for Fake News Detection via Reinforcement Learning [Paper](https://arxiv.org/abs/1912.12520) [Code](https://github.com/yaqingwang/WeFEND-AAAI20) AAAI 2020 Weakly Supervised Fake News Detection

Focusing on a specific scenario: News coming from Wechat with users' reports as noisy labels

Embracing on-policy RL to select highly confident labels

* Rumor Detection on Social Media with Bi-Directional Graph Convolutional Networks [Paper](https://ojs.aaai.org//index.php/AAAI/article/view/5393) [Code](https://github.com/TianBian95/BiGCN) AAAI 2020 rumor detection

A widely adopted baseline for rumor detection: BiGCN

Each event is represented as a tree-structured graph, with a news post as the root node and Twitter posts as the child nodes.

Adopting root-enhanced GCN as the encoder

Adopting DropEdge to enhance the robustness

* ReCOVery: A Multimodal Repository for COVID-19 News Credibility Research [Paper](https://arxiv.org/pdf/2006.05557.pdf) CIKM 2020 Benchmark

A multi-modal fake news detection benchmark for COVID-19

* Rumor Detection on Social Media with Graph Structured Adversarial Learning [Paper](https://www.ijcai.org/proceedings/2020/197) IJCAI 2020 Rumor Detection

Considering an adversarial setting(social interaction) for rumor detection

The adversarial attack is used as a regularizer to boost robust model.

The experiment part doesn't include a robustness test 

* Defending Against Neural Fake News [Paper](https://arxiv.org/abs/1905.12616) [Code](https://rowanzellers.com/grover) NIPS 2019 Fake News Generation

A fake news GPT that also utilizes CommonCrawl to achieve vast amounts of unlabeled data

* Jointly embedding the local and global relations of heterogeneous graph for rumor detection [Paper](https://arxiv.org/abs/1909.04465) [Code](https://github.com/chunyuanY/RumorDetection) ICDM 2019 Rumor Detection

A drawback of the tree-structured graph adopted by BiGCN: It ignores the relationship across different news posts

This paper adopts hierarchical attention to capture both local and global relations for rumors.

first-level attention: QKV attention to capture the inner-post relationship for retweets

second-level attention: a GAT to capture structural relationships among cross-post relationships

The performance is great, even better than some later work 

* defend: Explainable fake news detection [Paper](https://pike.psu.edu/publications/kdd19.pdf) [Code](https://github.com/cuilimeng/dEFEND-web) KDD 2019 fake news detection

An important baseline for context-based fake news detection

Adopting co-attention to capture the correlation between texts and comments

* Detect Rumor and Stance Jointly by Neural Multi-task Learning [Paper](https://dl.acm.org/doi/pdf/10.1145/3184558.3188729) WWW 2018 Multi-task learning
Learning models for rumor detection and stance classification at the same time

GRU-based encoder

a shared layer + task-specific layers
