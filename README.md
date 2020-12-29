# AISecMatrix

This is the github repository for ongoing project --- AISecMatrix （AI安全威胁矩阵）.



1. [Abstract](#1.Abstract)

2. [Environment Access](#2.Environment Access) 

* 2.1 [Dependent Software Attack](##2.1.Dependent Software Attack)
* 2.2 [Malicious Access to Docker](##2.2.Malicious Access to Docker)
* 2.3 [Hardware Backdoor Attack](##2.3.Hardware Backdoor Attack)
* 2.4 [Supply Chains Attack](##2.4.Supply Chains Attack)

----

1. [Abstract](#1.Abstract)

----





## 1.Abstract

In the past few years, AI techniques have gained wide application in a number of fields, including Image Processing, Speech Recognition, Natural Language Processing, etc. Hence, in security critical application senarios, the security issues of AI techniques have increasingly become the focus of research community and industrial corporations. Besides their performance, engineers and users should also take the security problems of the AI systems into account, and ensure the satefy of AI models in different business scenarios, avoiding serious consequences induced by malicious control, influence, fraud, faults and privacy disclosure.

To provide developers and users a better guidance on the security issues of AI systems, this report aims to release a framework **(Figure. 1)** to illusrate the attack process and specific attack techniques from the adversaries' perspectives, based on ATT&CK paradigm, which is already relatively mature in the network security domain.  Understanding and identifying these techniques is helpful for AI developers and maintainers to realize potential risk of AI systems during the overall life cycle and the corresponding solutions, providing essential technical guarantee for the application and deployment of AI systems.

![Figure. 1](img/1.png)









## 2.Environment Access





### 2.1.Dependent Software Attack

Machine Learning systems depend on the support from various bottom software frameworks and libraries. Security vulnerabilities hidden in these dependent softwares may pose serious threats to the integral safety of AI systems. Currently, there are various Deep Learning frameworks available, e.g. TensorFlow, Pytorch, etc. These software frameworks will further interact with hundreds of third-party dynamic libraries, including Numpy, OpenCV, etc. Security problems in these components will also severely threaten those systems based on Deep Learning framework. So far, a number of security bugs in dependent libraries of Deep Learning framework have already been reported, including Memory Overflow, Null Pointer Reference, Integer Overflow, Denial-of-Service Attack, etc. **[1,2]** For instance, in the past few years, in OpenCV library, which is one of the most widely used libraries in Deep Learning programs, several security bugs were discovered (especially the CVE-2019-5063 and CVE-2019-5064 reported at the end of 2019, which are quite severe). Based on these bugs, attackers can directly use traditional attack techniques **[3,4]** to bring about the threats of Arbitrary Code Execution to the application of AI systems. Besides, as shown in **Figure. 2**, these dependent libraries also have other types of bugs, potentially inducing Denial-of-Service Attack, Heap Overflow, Integer Overflow, etc., which will also pose severe threats to the safety of the systems.

Meanwhile,  attackers can also inject malicious codes into the model files used by Deep Learning frameworks (e.g. Tensorflow, Pytorch) **[6]**, which makes the common usage of third-party pretrained models also a risky behavior. These cases give us the caution that --- with the security research going deeper, users and developers should be more safety conscious, to defend against continuously emerging attack techniques.

**<u>Tips for defense:</u>** seasonably check and update the reported security bugs in dependent softwares and libraries, avoding damage to AI systems induced by bugs exploitation.

<img src="img/2-1-1.png" alt="Figure. 2" width="40%" /><img src="img/2-1-2.png" alt="Figure. 2" width="50%" />





### 2.2.Malicious Access to Docker

Machine Learning tasks can be deployed in Kubernetes clusters via KubeFlow framework **[5]**. Since usually the computation nodes for ML tasks have strong calculation capability, these nodes are thus becoming ideal attack targets for adversaries. For instance, attackers may hijack these ML tasks nodes and exploit them for mining. **[6]** One example is, in June 2020, Azure Security Center at Microsoft issued one warning after they detected malicious mining programs installed in Kuberflow by attackers. This security program was induced by improper configurations --- some users modified the default settings for panel access, changing Istio service into Load-Balancer, for convinient access. Such improper configurations made the service public for Internet, so that attackers could access the panel and deploy backdoor containers in the clusters via various methods. For example, with space search engines such as Shodan, Fofa, etc., adversaries can discover Kubernets exposed in public network and thus gaining opportunities to execute malicious codes **[5]**. As shown in **Figure. 3**, adversaries can complete the attacks by loading customized malicious Jupyter images, during the creation of Juputer application services in Kubeflow. Meanwhile, attackers can also directly deploy malicious containers via inserting additional python codes in Jupyter, which may further enlarge the attackers' accessibility to critical data/codes and even harm the integral security of the ML models.

**<u>Tips for defense:</u>** Developers and maintainers should be familiar with containers' common application scenarios and corresponding defensive techniques. As for relevent techniques, we refer interested readers to the Kubernetes threatening models **[8]** released by Microsoft. 

<img src="img/2-2-1.png" alt="Figure. 3" width="50%" /> <img src="img/2-2-2.png" alt="Figure. 3" width="33%" />





### 2.3.Hardware Backdoor Attack

Hardware Backdoors Attacks (also known as Hardware Trojans Attacks) can take place during the trained models being deployed in hardware devices, where adversaries may insert backdoors into the deployed models by making very slight modificaiton to hardware components, e.g. Lookup Table. Those models that contain backdoors can still operate normally on common cases, however, malicious behaviors could be triggered in certain preseted senarios, resulting in stealthy and severe theats.

Mordern integrated circuits usually contain third-party IP cores, which are commonly adopted as integrated modules for swift deployment. Such commonly used and modularized mechanisms enable hardware attackers to design Trojans for certain IP cores and thus correspondinly affecting a substantial number of hardware devices which use these modules. For instance, as illusrated in **[9]**, one can bring a neural network model to always making incorrect predictions by only modifying 0.03% of the model's parameteres during hardware deployment stage. **[10]** further showed that, by merely flipping 13 bits of a model with 93M bits, a ImageNet classifier with 70% accuracy could be reduced to a random classifier. Recently, **[11]** proposed Sequence Triggered Hardware Trojan for neural networks, which can totally invalidate a classifier, once a certain sequence of normal images are input into the model, as shown in **Figure. 4**.

So far, Hardware Backdoor Attack is still a newly emerging research area, and existing research study on this area is limited. However, in real application scenarios, this type of attack poses a severe threat. For example, attackers can inject backdoors into vision system of an autonomous driving car in the form of hardware Trojans, and the life safety of passengers may be seriously threatened if the backdoors are triggered. Note that, existing backdoors injection usually can only be  implemented by the models' owners. It would be very valuable to study the backdoors injection from outside invaders, since it's a more risky attack scenarios.

<img src="img/2-3-1.png" alt="Figure. 4" width="50%" />





### 2.4.Supply Chains Attack

As shown in **Figure. 5**, attackers can perform Supply Chains Attack in multiple ways, e.g. exploiting open source platform to release malicious pretrained models, constructing backdoors by controlling or modifying software and hardware platforms.

For instance, attackers can inject malicious instructions into model files by exploiting the security bugs Numpy CVE-2019-6446. Researchers already found that, when model loading functions like "torch.load" are executed, this security bug can be triggered, resulting in execution of malicious instructions. Since the execution of these malicious instructions will not affect the common usage of the models, such attacks are very stealthy. As shown in **Figure. 6**, using this bug, attackers can inject instructions into the model file (e.g. binary model file saved by Pytorch), and the "Calculator" program is launched when the model file is loaded. Similarly, attackers can also download and execute Trojan instructions in this way **[12]**, which makes this type of attacks very threatening.

**<u>Tips for defense:</u>** Make sure the srouce of model files are trustworthy before loading them, and be cautious in third-party model files.

<img src="img/2-4-1.png" alt="Figure. 5" width="50%" />![Figure. 5](img/2-4-2.png)

<img src="img/2-4-3.png" alt="Figure. 5" width="67%" />

