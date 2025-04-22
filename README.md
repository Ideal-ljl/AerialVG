# AerialVG

[![HuggingFace](https://img.shields.io/badge/Dataset-Huggingface-F8D44D.svg)](https://huggingface.co/datasets/IPEC-COMMUNITY/AerialVG)
[![Arxiv](https://img.shields.io/badge/Arxiv-Preprint-A42C24.svg)](https://arxiv.org/abs/2504.07836）
## Introduction
![cover](images/AerialVG.png)
Visual grounding (VG) aims to localize target objects in an image based on natural language descriptions. In this paper, we propose AerialVG, a new task focusing on visual grounding from aerial views. Compared to traditional VG, AerialVG poses new challenges, \emph{e.g.}, appearance-based grounding is insufficient to distinguish among multiple visually similar objects, and positional relations should be emphasized. Besides, existing VG models struggle when applied to aerial imagery, where high-resolution images cause significant difficulties. 
To address these challenges, we introduce the first AerialVG dataset, consisting of 5K real-world aerial images, 50K manually annotated descriptions, and 103K objects. Particularly, each annotation in AerialVG dataset contains multiple target objects annotated with relative spatial relations, requiring models to perform comprehensive spatial reasoning.
Furthermore, we propose an innovative model especially for the AerialVG task, where a Hierarchical Cross-Attention is devised to focus on target regions, and a Relation-Aware Grounding module is designed to infer positional relations. Experimental results validate the effectiveness of our dataset and method, highlighting the importance of spatial reasoning in aerial visual grounding.

Dataset has been released in HF🤗.The code is coming soon.
