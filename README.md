# AISecMatrix

1. [Abstract](#abstract)

2. [Environment Access](#environment-access) 

* 2.1 [Dependent Software Attack](#dependent-software-attack)
* 2.2 [Malicious Access to Docker](#malicious-access-to-docker)
* 2.3 [Hardware Backdoor Attack](#hardware-backdoor-attack)
* 2.4 [Supply Chains Attack](#supply-chains-attack)

<br><br>

[Reference](#reference)

[Go To End](#end)

----

This is the github repository for ongoing project --- AISecMatrix （AI安全威胁矩阵）.

<br><br><br><br>

<span id = "abstract"></span>

## 1. Abstract

In the past few years, AI techniques have gained wide application in a number of fields, including Image Processing, Speech Recognition, Natural Language Processing, etc. Hence, in security critical application senarios, the security issues of AI techniques have increasingly become the focus of research community and industrial corporations. Besides their performance, engineers and users should also take the security problems of the AI systems into account, and ensure the satefy of AI models in different business scenarios, avoiding serious consequences induced by malicious control, influence, fraud, faults and privacy disclosure.

To provide developers and users a better guidance on the security issues of AI systems, this report aims to release a framework **([Figure. 1](#figure-1))** to illusrate the attack process and specific attack techniques from the adversaries' perspectives, based on ATT&CK paradigm, which is already relatively mature in the network security domain.  Understanding and identifying these techniques is helpful for AI developers and maintainers to realize potential risk of AI systems during the overall life cycle and the corresponding solutions, providing essential technical guarantee for the application and deployment of AI systems.

**<span id = "figure-1">*<u>Figure-1</u>*</span>**

![Figure. 1](img/1.png)

<br><br><br><br>

<span id = "environment-access"></span>

## 2. Environment Access

<br>

<br>

<span id = "dependent-software-attack"></span>

### 2.1 Dependent Software Attack

Machine Learning systems depend on the support from various bottom software frameworks and libraries. Security vulnerabilities hidden in these dependent softwares may pose serious threats to the integral safety of AI systems. Currently, there are various Deep Learning frameworks available, e.g. TensorFlow, Pytorch, etc. These software frameworks will further interact with hundreds of third-party dynamic libraries, including Numpy, OpenCV, etc. Security problems in these components will also severely threaten those systems based on Deep Learning framework. So far, a number of security bugs in dependent libraries of Deep Learning framework have already been reported, including Memory Overflow, Null Pointer Reference, Integer Overflow, Denial-of-Service Attack, etc. **[[1](#ref-1),[2](#ref-2)]** For instance, in the past few years, in OpenCV library, which is one of the most widely used libraries in Deep Learning programs, several security bugs were discovered (especially the CVE-2019-5063 and CVE-2019-5064 reported at the end of 2019, which are quite severe). Based on these bugs, attackers can directly use traditional attack techniques **[[3](#ref-3),[4](#ref-4)]** to bring about the threats of Arbitrary Code Execution to the application of AI systems. Besides, as shown in **[Figure. 2](#figure-2)**, these dependent libraries also have other types of bugs, potentially inducing Denial-of-Service Attack, Heap Overflow, Integer Overflow, etc., which will also pose severe threats to the safety of the systems.

Meanwhile,  attackers can also inject malicious codes into the model files used by Deep Learning frameworks (e.g. Tensorflow, Pytorch) **[[6](#ref-6)]**, which makes the common usage of third-party pretrained models also a risky behavior. These cases give us the caution that --- with the security research going deeper, users and developers should be more safety conscious, to defend against continuously emerging attack techniques.

**<u>Tips for defense:</u>** seasonably check and update the reported security bugs in dependent softwares and libraries, avoding damage to AI systems induced by bugs exploitation.

**<span id = "figure-2">*<u>Figure-2</u>*</span>**

<img src="img/2-1-1.png" alt="Figure. 2" width="40%" /><img src="img/2-1-2.png" alt="Figure. 2" width="50%" />

<br>

<br>

<span id = "malicious-access-to-docker"></span>

### 2.2 Malicious Access to Docker

Machine Learning tasks can be deployed in Kubernetes clusters via KubeFlow framework **[[5](#ref-5)]**. Since usually the computation nodes for ML tasks have strong calculation capability, these nodes are thus becoming ideal attack targets for adversaries. For instance, attackers may hijack these ML tasks nodes and exploit them for mining. **[[6](#ref-6)]** One example is, in June 2020, Azure Security Center at Microsoft issued one warning after they detected malicious mining programs installed in Kuberflow by attackers. This security program was induced by improper configurations --- some users modified the default settings for panel access, changing Istio service into Load-Balancer, for convinient access. Such improper configurations made the service public for Internet, so that attackers could access the panel and deploy backdoor containers in the clusters via various methods. For example, with space search engines such as Shodan, Fofa, etc., adversaries can discover Kubernets exposed in public network and thus gaining opportunities to execute malicious codes **[[5](#ref-5)]**. As shown in **[Figure. 3](#figure-3)**, adversaries can complete the attacks by loading customized malicious Jupyter images, during the creation of Juputer application services in Kubeflow. Meanwhile, attackers can also directly deploy malicious containers via inserting additional python codes in Jupyter, which may further enlarge the attackers' accessibility to critical data/codes and even harm the integral security of the ML models.

**<u>Tips for defense:</u>** Developers and maintainers should be familiar with containers' common application scenarios and corresponding defensive techniques. As for relevent techniques, we refer interested readers to the Kubernetes threatening models **[[8](#ref-8)]** released by Microsoft. 

**<span id = "figure-3">*<u>Figure-3</u>*</span>**

<img src="img/2-2-1.png" alt="Figure. 3" width="50%" /> <img src="img/2-2-2.png" alt="Figure. 3" width="33%" />

<br>

<br>

<span id = "hardware-backdoor-attack"></span>

### 2.3 Hardware Backdoor Attack

Hardware Backdoors Attacks (also known as Hardware Trojans Attacks) can take place during the trained models being deployed in hardware devices, where adversaries may insert backdoors into the deployed models by making very slight modificaiton to hardware components, e.g. Lookup Table. Those models that contain backdoors can still operate normally on common cases, however, malicious behaviors could be triggered in certain preseted senarios, resulting in stealthy and severe theats.

Mordern integrated circuits usually contain third-party IP cores, which are commonly adopted as integrated modules for swift deployment. Such commonly used and modularized mechanisms enable hardware attackers to design Trojans for certain IP cores and thus correspondinly affecting a substantial number of hardware devices which use these modules. For instance, as illusrated in **[[9](#ref-9)]**, one can bring a neural network model to always making incorrect predictions by only modifying 0.03% of the model's parameteres during hardware deployment stage. **[[10](#ref-10)]** further showed that, by merely flipping 13 bits of a model with 93M bits, a ImageNet classifier with 70% accuracy could be reduced to a random classifier. Recently, **[[11](#ref-11)]** proposed Sequence Triggered Hardware Trojan for neural networks, which can totally invalidate a classifier, once a certain sequence of normal images are input into the model, as shown in **[Figure. 4](#figure-4)**.

So far, Hardware Backdoor Attack is still a newly emerging research area, and existing research study on this area is limited. However, in real application scenarios, this type of attack poses a severe threat. For example, attackers can inject backdoors into vision system of an autonomous driving car in the form of hardware Trojans, and the life safety of passengers may be seriously threatened if the backdoors are triggered. Note that, existing backdoors injection usually can only be  implemented by the models' owners. It would be very valuable to study the backdoors injection from outside invaders, since it's a more risky attack scenarios.

**<span id = "figure-4">*<u>Figure-4</u>*</span>**

<img src="img/2-3-1.png" alt="Figure. 4" width="50%" />

<br>

<br>

<span id = "supply-chains-attack"></span>

### 2.4 Supply Chains Attack

As shown in **[Figure. 5](#figure-5)**, attackers can perform Supply Chains Attack in multiple ways, e.g. exploiting open source platform to release malicious pretrained models, constructing backdoors by controlling or modifying software and hardware platforms.

For instance, attackers can inject malicious instructions into model files by exploiting the security bugs Numpy CVE-2019-6446. Researchers already found that, when model loading functions like "torch.load" are executed, this security bug can be triggered, resulting in execution of malicious instructions. Since the execution of these malicious instructions will not affect the common usage of the models, such attacks are very stealthy. As shown in **[Figure. 6](#figure-6)**, using this bug, attackers can inject instructions into the model file (e.g. binary model file saved by Pytorch), and the "Calculator" program is launched when the model file is loaded. Similarly, attackers can also download and execute Trojan instructions in this way **[[12](#ref-12)]**, which makes this type of attacks very threatening.

**<u>Tips for defense:</u>** Make sure the srouce of model files are trustworthy before loading them, and be cautious in third-party model files.

**<span id = "figure-5">*<u>Figure-5</u>*</span>**

<img src="img/2-4-1.png" alt="Figure. 5" width="50%" />![Figure. 5](img/2-4-2.png)

**<span id = "figure-6">*<u>Figure-6</u>*</span>**

<img src="img/2-4-3.png" alt="Figure. 5" width="67%" />

<br><br><br><br>

<span id = "reference"></span>

## Reference

<span id = "ref-1">[1] Q. Xiao, K. Li, D. Zhang, and W. Xu, “Security risks in deep learning implementa-tions,” in IEEE S&P Workshop, 2018.</span>

<span id = "ref-2">[2] https://www.cvedetails.com/vulnerability-list/vendor_id-1224/product_id-53738/Google-Tensorflow.html.</span>

<span id = "ref-3">[3] https://www.securityweek.com/serious-vulnerabilities-patched-opencv-computer-vision-library.</span>

<span id = "ref-4">[4] https://www.secpod.com/blog/opencv-buffer-overflow-vulnerabilities-jan-2020/.</span>

<span id = "ref-5">[5] https://www.microsoft.com/security/blog/2020/06/10/misconfigured-kubeflow-workloads-are-a-security-risk.</span>

<span id = "ref-6">[6] https://security.tencent.com/index.php/blog/msg/130.</span>

<span id = "ref-7">[7] https://www.kubeflow.org/docs/notebooks/setup/.</span>

<span id = "ref-8">[8] https://www.microsoft.com/security/blog/2020/04/02/attack-matrix-kubernetes/.</span>

<span id = "ref-9">[9] J. Clements and Y. Lao, “Hardware trojan attacks on neural networks”, arXiv preprintarXiv:1806.05768, 2018.</span>

<span id = "ref-10">[10] A. S. Rakin, Z. He, and D. Fan, “Bit-flip attack: Crushing neural network with pro-gressive bit search,” in ICCV, 2019.</span>

<span id = "ref-11">[11] Z. Liu, J. Ye, X. Hu, H. Li, X. Li, and Y. Hu, “Sequence triggered hardware trojan inneural network accelerator,” in VTS, 2020</span>

<span id = "ref-12">[12] nEINEI, ““黑”掉神经网络：通过逆向模型文件来重构模型后门,” in XFocus Infor-mation Security Conference, 2020.</span>

<span id = "ref-13">[13] C. Zhu, W. R. Huang, A. Shafahi, H. Li, G. Taylor, C. Studer, and T. Gold-stein, “Transferable clean-label poisoning attacks on deep neural nets,” arXiv preprintarXiv:1905.05897, 2019.</span>

<span id = "ref-14">[14] A. Shafahi, W. R. Huang, M. Najibi, O. Suciu, C. Studer, T. Dumitras, and T. Gold-stein, “Poison frogs! targeted clean-label poisoning attacks on neural networks,” in NeurIPS, 2018.</span>

<span id = "ref-15">[15] H. Xiao, B. Biggio, B. Nelson, H. Xiao, C. Eckert, and F. Roli, “Support vectormachines under adversarial label contamination,” Neurocomputing, vol. 160, pp. 53–62, 2015.</span>

<span id = "ref-16">[16] B. Biggio, B. Nelson, and P. Laskov, “Poisoning attacks against support vector ma-chines,”arXiv preprint arXiv:1206.6389, 2012.</span>

<span id = "ref-17">[17] S. Mei and X. Zhu, “Using machine teaching to identify optimal training-set attackson machine learners.” in AAAI, 2015.</span>

<span id = "ref-18">[18] J. Feng, Q.-Z. Cai, and Z.-H. Zhou, “Learning to confuse: Generating training timeadversarial data with auto-encoder,” in NeurIPS, 2019.</span>

<span id = "ref-19">[19] H. Chacon, S. Silva, and P. Rad, “Deep learning poison data attack detection,” in ICTAI, 2019.

<span id = "ref-20">[20] Y. Chen, Y. Mao, H. Liang, S. Yu, Y. Wei, and S. Leng, “Data poison detectionschemes for distributed machine learning,” IEEE Access, vol. 8, pp. 7442–7454, 2019.</span>

<span id = "ref-21">[21] Y. Li, B. Wu, Y. Jiang, Z. Li, and S.-T. Xia, “Backdoor learning: A survey,” arXivpreprint arXiv:2007.08745, 2020.</span>

<span id = "ref-22">[22] T. Gu, K. Liu, B. Dolan-Gavitt, and S. Garg, “Badnets: Evaluating backdooringattacks on deep neural networks,”IEEE Access, vol. 7, pp. 47230–47244, 2019.</span>

<span id = "ref-23">[23] A. Saha, A. Subramanya, and H. Pirsiavash, “Hidden trigger backdoor attacks,” in AAAI, 2020.</span>

<span id = "ref-24">[24] S. Zhao, X. Ma, X. Zheng, J. Bailey, J. Chen, and Y.-G. Jiang, “Clean-label backdoorattacks on video recognition models,” in CVPR, 2020.</span>

<span id = "ref-25">[25] X. Chen, C. Liu, B. Li, K. Lu, and D. Song, “Targeted backdoor attacks on deeplearning systems using data poisoning,” arXiv preprint arXiv:1712.05526, 2017.</span>

<span id = "ref-26">[26] J. Dai, C. Chen, and Y. Li, “A backdoor attack against lstm-based text classificationsystems,” IEEE Access, vol. 7, pp. 138872–138878, 2019.</span>

<span id = "ref-27">[27] E. Bagdasaryan, A. Veit, Y. Hua, D. Estrin, and V. Shmatikov, “How to backdoorfederated learning,” in AISTATS, 2020.</span>

<span id = "ref-28">[28] K. Kurita, P. Michel, and G. Neubig, “Weight poisoning attacks on pre-trained mod-els,” in ACL, 2020.</span>

<span id = "ref-29">[29] B. Wang, X. Cao, N. Z. Gong,et al., “On certifying robustness against backdoorattacks via randomized smoothing,” in CVPR Workshop, 2020.</span>

<span id = "ref-30">[30] M. Weber, X. Xu, B. Karlas, C. Zhang, and B. Li, “Rab: Provable robustness againstbackdoor attacks,”arXiv preprint arXiv:2003.08904, 2020.</span>

<span id = "ref-31">[31] Y. Liu, Y. Xie, and A. Srivastava, “Neural trojans,” in ICCD, 2017.</span>

<span id = "ref-32">[32] B. G. Doan, E. Abbasnejad, and D. C. Ranasinghe, “Februus: Input purificationdefense against trojan attacks on deep neural network systems,” in arXiv: 1908.03369,2019.</span>

<span id = "ref-33">[33] Y. Li, T. Zhai, B. Wu, Y. Jiang, Z. Li, and S. Xia, “Rethinking the trigger of backdoorattack,”arXiv preprint arXiv:2004.04692, 2020.</span>

<span id = "ref-34">[34] K. Liu, B. Dolan-Gavitt, and S. Garg, “Fine-pruning: Defending against backdooringattacks on deep neural networks,” in RAID, 2018.</span>

<span id = "ref-35">[35] P. Zhao, P.-Y. Chen, P. Das, K. N. Ramamurthy, and X. Lin, “Bridging mode con-nectivity in loss landscapes and adversarial robustness,” in ICLR, 2020.</span>

<span id = "ref-36">[36] B. Wang, Y. Yao, S. Shan, H. Li, B. Viswanath, H. Zheng, and B. Y. Zhao, “Neuralcleanse: Identifying and mitigating backdoor attacks in neural networks,” in IEEES&P, 2019.</span>

<span id = "ref-37">[37] X. Qiao, Y. Yang, and H. Li, “Defending neural backdoors via generative distributionmodeling,” in NeurIPS, 2019.</span>

<span id = "ref-38">[38] H. Chen, C. Fu, J. Zhao, and F. Koushanfar, “Deepinspect: A black-box trojan detec-tion and mitigation framework for deep neural networks.” in IJCAI, 2019.</span>

<span id = "ref-39">[39] S. Kolouri, A. Saha, H. Pirsiavash, and H. Hoffmann, “Universal litmus patterns:Revealing backdoor attacks in cnns,” in CVPR, 2020.</span>

<span id = "ref-40">[40] S. Huang, W. Peng, Z. Jia, and Z. Tu, “One-pixel signature: Characterizing cnn modelsfor backdoor detection,” in ECCV, 2020.</span>

<span id = "ref-41">[41] R. Wang, G. Zhang, S. Liu, P.-Y. Chen, J. Xiong, and M. Wang, “Practical detectionof trojan neural networks: Data-limited and data-free cases,” in ECCV, 2020.</span>

<span id = "ref-42">[42] B. Tran, J. Li, and A. Madry, “Spectral signatures in backdoor attacks,” in NeurIPS,2018.</span>

<span id = "ref-43">[43] B. Chen, W. Carvalho, N. Baracaldo, H. Ludwig, B. Edwards, T. Lee, I. Molloy,and B. Srivastava, “Detecting backdoor attacks on deep neural networks by activationclustering,” in AAAI Workshop, 2018.</span>

<span id = "ref-44">[44] Y. Gao, C. Xu, D. Wang, S. Chen, D. C. Ranasinghe, and S. Nepal, “Strip: A defenceagainst trojan attacks on deep neural networks,” in ACSAC, 2019.</span>

<span id = "ref-45">[45] M. Du, R. Jia, and D. Song, “Robust anomaly detection and backdoor attack detectionvia differential privacy,” in ICLR, 2020.</span>

<span id = "ref-46">[46] S. Hong, V. Chandrasekaran, Y. Kaya, T. Dumitraş, and N. Papernot, “On the effec-tiveness of mitigating data poisoning attacks with gradient shaping,”arXiv preprintarXiv:2002.11497, 2020.</span>

<span id = "ref-47">[47] S. B. Venkatakrishnan, S. Gupta, H. Mao, M. Alizadeh,et al., “Learning generalizabledevice placement algorithms for distributed machine learning,” in NeurIPS, 2019.</span>

<span id = "ref-48">[48] L. Zhu, Z. Liu, and S. Han, “Deep leakage from gradients,” in NeurIPS, 2019.</span>

<span id = "ref-49">[49] K. Grosse, T. A. Trost, M. Mosbach, M. Backes, and D. Klakow, “Adversar-ial initialization–when your network performs the way i want,”arXiv preprintarXiv:1902.03020, 2019.</span>

<span id = "ref-50">[50] P. Blanchard, R. Guerraoui, J. Stainer,et al., “Machine learning with adversaries:Byzantine tolerant gradient descent,” in NeurIPS, 2017.</span>

<span id = "ref-51">[51] Z. Sun, P. Kairouz, A. T. Suresh, and H. B. McMahan, “Can you really backdoorfederated learning?”arXiv preprint arXiv:1911.07963, 2019.</span>

<span id = "ref-52">[52] C. Xie, K. Huang, P.-Y. Chen, and B. Li, “Dba: Distributed backdoor attacks againstfederated learning,” in ICLR, 2019.</span>

<span id = "ref-53">[53] H. Yin, P. Molchanov, J. M. Alvarez, Z. Li, A. Mallya, D. Hoiem, N. K. Jha, andJ. Kautz, “Dreaming to distill: Data-free knowledge transfer via deepinversion,” in CVPR, 2020.</span>

<span id = "ref-54">[54] I. J. Goodfellow, J. Shlens, and C. Szegedy, “Explaining and harnessing adversarialexamples,” STAT, vol. 1050, p. 20, 2015.</span>

<span id = "ref-55">[55] Y. Fan, B. Wu, T. Li, Y. Zhang, M. Li, Z. Li, and Y. Yang, “Sparse adversarial attackvia perturbation factorization,” in ECCV, 2020.</span>

<span id = "ref-56">[56] J. Bai, B. Chen, Y. Li, D. Wu, W. Guo, S.-t. Xia, and E.-h. Yang, “Targeted attackfor deep hashing based retrieval,” ECCV, 2020.</span>

<span id = "ref-57">[57] Y. Xu, B. Wu, F. Shen, Y. Fan, Y. Zhang, H. T. Shen, and W. Liu, “Exact adversarialattack to image captioning via structured output learning with latent variables,” in CVPR, 2019.</span>

<span id = "ref-58">[58] X. Chen, X. Yan, F. Zheng, Y. Jiang, S.-T. Xia, Y. Zhao, and R. Ji, “One-shotadversarial attacks on visual tracking with dual attention,” in CVPR, 2020.</span>

<span id = "ref-59">[59] C. Szegedy, V. Vanhoucke, S. Ioffe, J. Shlens, and Z. Wojna, “Rethinking the inceptionarchitecture for computer vision,” in CVPR, 2016.</span>

<span id = "ref-60">[60] J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, and L. Fei-Fei, “Imagenet: A large-scalehierarchical image database,” in CVPR, 2009.</span>

<span id = "ref-61">[61] Y. Guo, Z. Yan, and C. Zhang, “Subspace attack: Exploiting promising subspaces forquery-efficient black-box attacks,” in NeurIPS, 2019.</span>

<span id = "ref-62">[62] Y. Dong, H. Su, B. Wu, Z. Li, W. Liu, T. Zhang, and J. Zhu, “Efficient decision-basedblack-box adversarial attacks on face recognition,” in CVPR, 2019.</span>

<span id = "ref-63">[63] Z. Wei, J. Chen, X. Wei, L. Jiang, T.-S. Chua, F. Zhou, and Y.-G. Jiang, “Heuristicblack-box adversarial attacks on video recognition models.” in AAAI, 2020.</span>

<span id = "ref-64">[64] B. Ru, A. Cobb, A. Blaas, and Y. Gal, “Bayesopt adversarial attack,” in ICLR, 2020.</span>

<span id = "ref-65">[65] L. Meunier, J. Atif, and O. Teytaud, “Yet another but more efficient black-box ad-versarial attack: tiling and evolution strategies,”arXiv preprint arXiv:1910.02244,2019.</span>

<span id = "ref-66">[66] P. Zhao, S. Liu, P.-Y. Chen, N. Hoang, K. Xu, B. Kailkhura, and X. Lin, “On thedesign of black-box adversarial examples by leveraging gradient-free optimization andoperator splitting method,” in ICCV, 2019.</span>

<span id = "ref-67">[67] A. Al-Dujaili and U.-M. O’Reilly, “Sign bits are all you need for black-box attacks,”in ICLR, 2019.</span>

<span id = "ref-68">[68] N. Papernot, P. McDaniel, I. Goodfellow, S. Jha, Z. B. Celik, and A. Swami, “Practicalblack-box attacks against machine learning,” in ASIACCS, 2017.</span>

<span id = "ref-69">[69] Z. Huang and T. Zhang, “Black-box adversarial attack with transferable model-basedembedding,”arXiv preprint arXiv:1911.07140, 2019.</span>

<span id = "ref-70">[70] Y. Feng, B. Wu, Y. Fan, Z. Li, and S. Xia, “Efficient black-box adversarialattack guided by the distribution of adversarial perturbations,” arXiv preprintarXiv:2006.08538, 2020.</span>

<span id = "ref-71">[71] P.-Y. Chen, H. Zhang, Y. Sharma, J. Yi, and C.-J. Hsieh, “Zoo: Zeroth order opti-mization based black-box attacks to deep neural networks without training substitutemodels,” in Proceedings of the 10th ACM Workshop on Artificial Intelligence and Se-curity, 2017, pp. 15–26.</span>

<span id = "ref-72">[72] M. Andriushchenko, F. Croce, N. Flammarion, and M. Hein, “Square attack: a query-efficient black-box adversarial attack via random search,” ECCV, 2020.</span>

<span id = "ref-73">[73] A. Krizhevsky, G. Hinton,et al., “Learning multiple layers of features from tiny im-ages,” 2009.</span>

<span id = "ref-74">[74] C. Szegedy, W. Zaremba, I. Sutskever, J. Bruna, D. Erhan, I. J. Goodfellow, andR. Fergus, “Intriguing properties of neural networks,” in ICLR, 2014.</span>

<span id = "ref-75">[75] Y. Gong and C. Poellabauer, “Crafting adversarial examples for speech paralinguisticsapplications,”arXiv preprint arXiv:1711.03280, 2017.</span>

<span id = "ref-76">[76] C. Kereliuk, B. L. Sturm, and J. Larsen, “Deep learning and music adversaries,”IEEETransactions on Multimedia, vol. 17, no. 11, pp. 2059–2071, 2015.</span>

<span id = "ref-77">[77] M. Cisse, Y. Adi, N. Neverova, and J. Keshet, “Houdini: Fooling deep structuredprediction models,”arXiv preprint arXiv:1707.05373, 2017.</span>

<span id = "ref-78">[78] N. Carlini, P. Mishra, T. Vaidya, Y. Zhang, M. Sherr, C. Shields, D. Wagner, andW. Zhou, “Hidden voice commands,” in USENIX, 2016.</span>

<span id = "ref-79">[79] G. Zhang, C. Yan, X. Ji, T. Zhang, T. Zhang, and W. Xu, “Dolphinattack: Inaudiblevoice commands,” in CCS, 2017.</span>

<span id = "ref-80">[80] N. Carlini and D. Wagner, “Audio adversarial examples: Targeted attacks on speech-to-text,” in IEEE S&P Workshop, 2018.</span>

<span id = "ref-81">[81] J. Li, S. Ji, T. Du, B. Li, and T. Wang, “Textbugger: Generating adversarial textagainst real-world applications,”arXiv preprint arXiv:1812.05271, 2018.</span>

<span id = "ref-82">[82] W. Fan, T. Derr, X. Zhao, Y. Ma, H. Liu, J. Wang, J. Tang, and Q. Li, “Attackingblack-box recommendations via copying cross-domain user profiles,”arXiv preprintarXiv:2005.08147, 2020.</span>

<span id = "ref-83">[83] S. Huang, N. Papernot, I. Goodfellow, Y. Duan, and P. Abbeel, “Adversarial attackson neural network policies,”arXiv preprint arXiv:1702.02284, 2017.</span>

<span id = "ref-84">[84] K. Eykholt, I. Evtimov, E. Fernandes, B. Li, A. Rahmati, C. Xiao, A. Prakash,T. Kohno, and D. Song, “Robust physical-world attacks on deep learning visual clas-sification,” in CVPR, 2018.</span>

<span id = "ref-85">[85] M. Sharif, S. Bhagavatula, L. Bauer, and M. K. Reiter, “Accessorize to a crime: Realand stealthy attacks on state-of-the-art face recognition,” in CCS, 2016.</span>

<span id = "ref-86">[86] A. Athalye, L. Engstrom, A. Ilyas, and K. Kwok, “Synthesizing robust adversarialexamples,” in ICML, 2018.</span>

<span id = "ref-87">[87] M. Sharif, S. Bhagavatula, L. Bauer, and M. K. Reiter, “A general framework foradversarial examples with objectives,”ACM Transactions on Privacy and Security(TOPS), vol. 22, no. 3, pp. 1–30, 2019.</span>

<span id = "ref-88">[88] Y. Zhao, H. Zhu, R. Liang, Q. Shen, S. Zhang, and K. Chen, “Seeing isn’tbelieving:  Practical adversarial attack against object detectors,”arXiv preprintarXiv:1812.10217, 2018.</span>

<span id = "ref-89">[89] L. Huang, C. Gao, Y. Zhou, C. Xie, A. L. Yuille, C. Zou, and N. Liu, “Universalphysical camouflage attacks on object detectors,” in CVPR, 2020.</span>

<span id = "ref-90">[90] Z. Kong, J. Guo, A. Li, and C. Liu, “Physgan: Generating physical-world-resilientadversarial examples for autonomous driving,” in CVPR, 2020.</span>

<span id = "ref-91">[91] R. Duan, X. Ma, Y. Wang, J. Bailey, A. K. Qin, and Y. Yang, “Adversarial camouflage:Hiding physical-world attacks with natural styles,” in CVPR, 2020.</span>

<span id = "ref-92">[92] Z. Wang, S. Zheng, M. Song, Q. Wang, A. Rahimpour, and H. Qi, “advpattern:Physical-world attacks on deep person re-identification via adversarially transformable patterns,” in ICCV, 2019.</span>

<span id = "ref-93">[93] T. Orekondy, B. Schiele, and M. Fritz, “Knockoff nets: Stealing functionality of black-box models,” in CVPR, 2019.</span>

<span id = "ref-94">[94] F. Tramèr, F. Zhang, A. Juels, M. K. Reiter, and T. Ristenpart, “Stealing machinelearning models via prediction apis,” in USENIX, 2016.</span>

<span id = "ref-95">[95] M. Juuti, S. Szyller, S. Marchal, and N. Asokan, “Prada: protecting against dnn modelstealing attacks,” in EuroS&P.  IEEE, 2019.</span>

<span id = "ref-96">[96] H. Yu, K. Yang, T. Zhang, Y.-Y. Tsai, T.-Y. Ho, and Y. Jin, “Cloudleak: Large-scaledeep learning models stealing through adversarial examples,” in NDSS, 2020.</span>

<span id = "ref-97">[97] https://github.com/Kayzaks/HackingNeuralNetworks.</span>

<span id = "ref-98">[98] https://github.com/Kayzaks/HackingNeuralNetworks/.</span>

<span id = "ref-99">[99] R. Stevens, O. Suciu, A. Ruef, S. Hong, M. Hicks, and T. Dumitraş, “Summon-ing demons: The pursuit of exploitable bugs in machine learning,”arXiv preprintarXiv:1701.04739, 2017.</span>

<span id = "ref-100">[100] D. Su, H. Zhang, H. Chen, J. Yi, P.-Y. Chen, and Y. Gao, “Is robustness the costof accuracy?–a comprehensive study on the robustness of 18 deep image classificationmodels,” in ECCV, 2018.</span>

<span id = "ref-101">[101] D. Rolnick and K. P. Kording, “Identifying weights and architectures of unknown relunetworks,”arXiv preprint arXiv:1910.00744, 2019.</span>

<span id = "ref-102">[102] S. Hong, M. Davinroy, Y. Kaya, S. N. Locke, I. Rackow, K. Kulda, D. Dachman-Soled,and T. Dumitraş, “Security analysis of deep neural networks operating in the presenceof cache side-channel attacks,”arXiv preprint arXiv:1810.03487, 2018.</span>

<span id = "ref-103">[103] S. Chen, P. Zhang, C. Sun, J. Cai, and X. Huang, “Generate high-resolution adversarialsamples by identifying effective features,”arXiv preprint arXiv:2001.07631, 2020.</span>

<span id = "ref-104">[104] C. Xie, J. Wang, Z. Zhang, Y. Zhou, L. Xie, and A. Yuille, “Adversarial examples forsemantic segmentation and object detection,” in ICCV, 2017.</span>

<span id = "ref-105">[105] J. Li, R. Ji, H. Liu, X. Hong, Y. Gao, and Q. Tian, “Universal perturbation attackagainst image retrieval,” in ICCV, 2019.</span>

<span id = "ref-106">[106] C. Sun, S. Chen, J. Cai, and X. Huang, “Type i attack for generative models,”arXivpreprint arXiv:2003.01872, 2020. </span>

<span id = "ref-107">[107] S. Tang, X. Huang, M. Chen, C. Sun, and J. Yang, “Adversarial attack type I: Cheatclassifiers by significant changes,”IEEE Transactions on Pattern Analysis and Machine Intelligence, 2019.</span>

<span id = "ref-108">[108] Y.-C. Lin, Z.-W. Hong, Y.-H. Liao, M.-L. Shih, M.-Y. Liu, and M. Sun, “Tac-tics of adversarial attack on deep reinforcement learning agents,”arXiv preprintarXiv:1703.06748, 2017.</span>

<span id = "ref-109">[109] S. Chen, F. He, X. Huang, and K. Zhang, “Attack on multi-node attention for objectdetection,”arXiv preprint arXiv:2008.06822, 2020.</span>

<span id = "ref-110">[110] S. Chen, Z. He, C. Sun, and X. Huang, “Universal adversarial attack on attention andthe resulting dataset damagenet,”arXiv preprint arXiv:2001.06325, 2020.</span>

<span id = "ref-111">[111] J. Han, X. Dong, R. Zhang, D. Chen, W. Zhang, N. Yu, P. Luo, and X. Wang, “Once aman: Towards multi-target attack via learning multi-target adversarial network once,” in ICCV, 2019.</span>

<span id = "ref-112">[112] J. Sun, T. Chen, G. Giannakis, and Z. Yang, “Communication-efficient distributedlearning via lazily aggregated quantized gradients,” in NeurIPS, 2019.</span>

<span id = "ref-113">[113] F. He, X. Huang, K. Lv, and J. Yang, “A communication-efficient distributed algorithmfor kernel principal component analysis,”arXiv preprint arXiv:2005.02664, 2020.</span>





<br><br><br><br>

<span id = "end">**End**</span>