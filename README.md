
# Teacher-Ensemble distillation based learned Label Interpolation

I propose a Semi Supervised classification approach using two Teacher models and a student model. The goal is to leverage both labeled and unlabeled data for improved classification performance. The methodology involves training two Teacher models on the labeled data and distilling their knowledge to a student model through knowledge distillation. 

The student model learns from the soft labels and logits of the Teacher models, enabling transfer of knowledge and improving generalization. To further enhance the learning process, a Meta lambda learning strategy is employed. A lambda head is introduced in the model, which takes the Teacher's logits as input and predicts a lambda value for label mixup. The lambda value determines the balance between the two Teacher models during training. 

The lambda head is optimized using a meta learning approach, allowing the model to dynamically adapt the weighting between Teachers based on their performance. This methodology enables the student model to effectively learn from the Teachers and achieve better classification accuracy on both labeled and unlabeled data.

This work was part of a BS Thesis project at the Indian Institute of Science Education Research Bhopal, under the supervision of Dr. Akshay Agarwal. Kindly go through my [thesis ](https://github.com/Dubeman/Teacher-Ensembling-based-learned-Label-Interpolation-TELI-SSL-Classification/blob/main/ManasDubey_19184_report.pdf) for a detailed analysis of this strategy and results on different benchmark datasets.












## Acknowledgements

 - [https://github.com/kekmodel/MPL-pytorch](https://github.com/kekmodel/MPL-pytorch)



## Deployment

To run the TELI model without meta learning lambda :-

```bash
  python3 main.py
```

To run the TELI model with meta learning lambda :-

```bash
  python3 main_meta.py
```

The above files will automatically download the datasets listed in args.dataset parser and initialize the appropriate student and teacher models in the default section for teacher_1 and teacher_2 . 

I have trained teacher models and saved their file paths.Attached below are the links to some of the pre trained teacher models and you have to store them in a directory. Also in the main function you have to provide the file paths for the teachers.

[Student Paths](https://drive.google.com/drive/folders/1-Glnq6g6JkjXrhg5RxWnZ2kce_Gt1XJD?usp=share_link)

[Teacher Paths](https://drive.google.com/drive/folders/1-SU52-TrsSLKUVPEIjd6K3DTftWxvqjF?usp=share_link)


Only the combination of teacher models which have been instantiated in the main parts ,with the appropriate dataset can be loaded to train the student models.
## 🔗 Links
[![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/manas-dubey-aba466234/)




