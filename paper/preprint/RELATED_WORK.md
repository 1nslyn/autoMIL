# Relation to prior work (draft note for the preprint)

> Drop-in paragraphs that stake the `autoMIL` name and separate it from the
> published neighbors and the same-named GitHub repo. BibTeX below.

## Prose

Recent frameworks have advanced multiple instance learning (MIL) for
computational pathology. PathBench-MIL \[Brussee et al., 2025\] automates
end-to-end MIL pipeline construction, covering preprocessing, feature
extraction, and aggregation, and standardizes evaluation across datasets. nnMIL
\[Luo et al., 2025\] offers a generalizable MIL framework that connects
patch-level foundation models to robust slide-level inference. Both systems fix
a pipeline and optimize configurations within it: the practitioner defines the
space of models, features, and hyperparameters, and the framework searches that
space.

autoMIL operates at a different level of automation. Instead of searching a
predefined grid, it automates the experimentation process itself. A coding agent
reads an existing MIL codebase, proposes and implements source-level changes,
executes them as isolated experiments, and records the outcomes in a persistent
experiment tree that informs later proposals. The unit of automation is the
research iteration rather than the hyperparameter, which lets autoMIL explore
modeling ideas that fall outside any fixed pipeline.

We also note a naming coincidence. An unpublished repository,
`frankkramer-lab/AutoMIL`, provides a conventional MIL training and evaluation
pipeline for whole-slide images. It shares the name but not the scope of our
work, and to our knowledge it has no associated publication. Throughout this
paper, autoMIL refers to the autonomous experimentation framework introduced
here.

## BibTeX

```bibtex
@article{brussee2025pathbenchmil,
  title   = {PathBench-MIL: A Comprehensive AutoML and Benchmarking Framework
             for Multiple Instance Learning in Histopathology},
  author  = {Brussee, Siemen and Valkema, Pieter A. and Weijer, Jurre A. J. and
             Doeleman, Thom and Schrader, Anne M. R. and Kers, Jesper},
  journal = {arXiv preprint arXiv:2512.17517},
  year    = {2025}
}

@article{luo2025nnmil,
  title   = {nnMIL: A Generalizable Multiple Instance Learning Framework for
             Computational Pathology},
  author  = {Luo, Xiangde and Xiang, Jinxi and Ji, Yuanfeng and Li, Ruijiang},
  journal = {arXiv preprint arXiv:2511.14907},
  year    = {2025}
}

@inproceedings{ilse2018attention,
  title     = {Attention-based Deep Multiple Instance Learning},
  author    = {Ilse, Maximilian and Tomczak, Jakub M. and Welling, Max},
  booktitle = {International Conference on Machine Learning (ICML)},
  year      = {2018}
}

@misc{kramerlab_automil,
  title        = {{AutoMIL}: Automated Machine Learning for Image Classification
                  in Whole-Slide Imaging with Multiple Instance Learning},
  author       = {{Kramer Lab}},
  howpublished = {\url{https://github.com/frankkramer-lab/AutoMIL}},
  note         = {Unpublished software repository},
  year         = {2025}
}
```

<!-- TODO(leo): confirm the arXiv IDs render to the final citations once the
     neighbors are camera-ready, and verify frankkramer-lab/AutoMIL is still
     unpublished at submission time. -->
