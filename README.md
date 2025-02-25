# Introduction
Welcome to the repository for Applications of AI (COMP 3000) Midterm!
The goal for this application(s) was to use the best combination of huggingface models to create a
fantasy styled recipe generator.

# Usage
You may need to install dependencies (see dependencies.py, cannot run that but those are the used libraries)
Explictly named:
PyTorch
TensorFlow
Transformers // See [Hugging Face](huggingface.co)

This must be updated, however you can run QwenTest.py with Python 3.11+ 

## Important files
QwenTest.py - Best model used so far (slightly costly)
testChef-2.py - Provided good inspiration for recipes, limited thoughts to only enhancing the creative prowess of recipes

Other files are either broken, or produce broken output.

## Questions or Concerns
Send them here: Stovez112@outlook.com

## Credits
Lots of debugging and prompt editing done by Chat-GPT (Reduced repetition and ambiguity)


### Models:

Ashikan/dut-recipe-generator
    Prof Ashika Naicker*,  Mr Shaylin Chetty,  Ms Riashnie Thaver*, Ms. Anjellah Reddy*, Dr. Evonne Shanita Singh*, Dr. Imana Pal*, Dr. Lisebo Mothepu*.

    *Durban University of Technology, Faculty of Applied Sciences, Department of Food and Nutrition,  Durban, South Africa

Qwen/Qwen2.5-3B-Instruct
        @misc{qwen2.5,
        title = {Qwen2.5: A Party of Foundation Models},
        url = {https://qwenlm.github.io/blog/qwen2.5/},
        author = {Qwen Team},
        month = {September},
        year = {2024}
    }

    @article{qwen2,
        title={Qwen2 Technical Report}, 
        author={An Yang and Baosong Yang and Binyuan Hui and Bo Zheng and Bowen Yu and Chang Zhou and Chengpeng Li and Chengyuan Li and Dayiheng Liu and Fei Huang and Guanting Dong and Haoran Wei and Huan Lin and Jialong Tang and Jialin Wang and Jian Yang and Jianhong Tu and Jianwei Zhang and Jianxin Ma and Jin Xu and Jingren Zhou and Jinze Bai and Jinzheng He and Junyang Lin and Kai Dang and Keming Lu and Keqin Chen and Kexin Yang and Mei Li and Mingfeng Xue and Na Ni and Pei Zhang and Peng Wang and Ru Peng and Rui Men and Ruize Gao and Runji Lin and Shijie Wang and Shuai Bai and Sinan Tan and Tianhang Zhu and Tianhao Li and Tianyu Liu and Wenbin Ge and Xiaodong Deng and Xiaohuan Zhou and Xingzhang Ren and Xinyu Zhang and Xipin Wei and Xuancheng Ren and Yang Fan and Yang Yao and Yichang Zhang and Yu Wan and Yunfei Chu and Yuqiong Liu and Zeyu Cui and Zhenru Zhang and Zhihao Fan},
        journal={arXiv preprint arXiv:2407.10671},
        year={2024}
    }

openai-community/gpt2
    @article{radford2019language,
        title={Language Models are Unsupervised Multitask Learners},
        author={Radford, Alec and Wu, Jeff and Child, Rewon and Luan, David and Amodei, Dario and Sutskever, Ilya},
        year={2019}
        }


google-t5/t5-base
google-t5/t5-3b
@article{2020t5,
  author  = {Colin Raffel and Noam Shazeer and Adam Roberts and Katherine Lee and Sharan Narang and Michael Matena and Yanqi Zhou and Wei Li and Peter J. Liu},
  title   = {Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer},
  journal = {Journal of Machine Learning Research},
  year    = {2020},
  volume  = {21},
  number  = {140},
  pages   = {1-67},
  url     = {http://jmlr.org/papers/v21/20-074.html}
}

google-bert/bert-base-uncased
@article{DBLP:journals/corr/abs-1810-04805,
  author    = {Jacob Devlin and
               Ming{-}Wei Chang and
               Kenton Lee and
               Kristina Toutanova},
  title     = {{BERT:} Pre-training of Deep Bidirectional Transformers for Language
               Understanding},
  journal   = {CoRR},
  volume    = {abs/1810.04805},
  year      = {2018},
  url       = {http://arxiv.org/abs/1810.04805},
  archivePrefix = {arXiv},
  eprint    = {1810.04805},
  timestamp = {Tue, 30 Oct 2018 20:39:56 +0100},
  biburl    = {https://dblp.org/rec/journals/corr/abs-1810-04805.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}


 