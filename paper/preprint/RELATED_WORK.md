# Relation to prior work (draft note for the preprint)

> Drop-in paragraphs that stake the `autoMIL` name and separate it from the
> published neighbors and the same-named GitHub repo. BibTeX below.

## Prose

### Autonomous ML experimentation

Language agents that edit and execute ML code are already an established
research direction. MLAgentBench \[Huang et al., 2024\] evaluates agents that
read and write files, execute code, inspect outputs, and iterate over ML
experiments. AIDE \[Jiang et al., 2025\] goes further by formulating ML
engineering as code optimization and trial-and-error as tree search over
candidate programs. AI Scientist-v2 \[Yamada et al., 2025\] uses progressive
agentic tree search across multiple ML domains, while AIRA \[Toledo et al.,
2025\] formalizes research agents as search policies over candidate-solution
graphs with improvement, debugging, memory, and crossover operators under
bounded, isolated execution.

These systems mean that autoMIL cannot claim source-level modification,
experiment trees, persistent memory, resource budgets, or autonomous research
iteration as individually new concepts. Its system claim is narrower: autoMIL
is a research-operations substrate for applying coding agents to existing
multi-file repositories, representing each experiment as a parent-addressed
source overlay that is reconstructed in an isolated git worktree and governed
by one result/provenance contract across execution backends.

### Auditable evaluation of research agents

Controlled and hidden evaluation also has direct precedents. RE-Bench \[Wijk et
al., 2024\] compares agents and human experts under matched time budgets and
releases agent trajectories. MLRC-Bench \[Zhang et al., 2025\] evaluates
repository-level code changes under fixed step/time budgets, saves intermediate
code snapshots, selects a candidate using development performance, and evaluates
the selected candidate on a held-out test. AIRA separates validation-guided
search from held-out evaluation; AIRA2 \[Hambardzumyan et al., 2026\] strengthens
this design with Hidden Consistent Evaluation and asynchronous isolated
execution. In medical imaging, AMID \[Liu et al., 2026\] combines autonomous
method development with explicit verification of validation protocols, metrics,
and prediction artifacts.

Accordingly, autoMIL does not claim to invent equal budgets, validation-only
selection, hidden tests, sandboxing, or trajectory provenance. Its protocol is
part of the C1 system contract: predeclared editable/protected source surfaces,
matched launched-attempt accounting across competing model lineages with
usable-result counts reported separately,
parent-linked multi-file diffs, and framework-mediated non-interference between
search and held-out certification. The paper's empirical contribution is then
to use that contract to ask a different question from agent/scaffold
leaderboards: whether published pathology-MIL rankings remain stable after each
lineage receives the same declared autonomous research opportunity.

### Computational-pathology MIL frameworks

Recent frameworks have also advanced multiple instance learning (MIL) for
computational pathology. PathBench-MIL \[Brussee et al., 2025\] automates
end-to-end MIL pipeline construction, covering preprocessing, feature
extraction, aggregation, and configured AutoML search. nnMIL \[Luo et al.,
2025\] offers a generalizable framework connecting patch-level foundation
models to robust slide-level inference. These works are essential MIL
infrastructure and benchmark comparators, but neither is treated here as the
closest prior system for agentic code-space research.

Relative to fixed-pipeline search, autoMIL allows the intervention to be a
source-code change rather than only a predeclared scalar or menu choice.
Relative to autonomous-research systems, its intended distinction is not the
existence of code editing or tree search, but the controlled comparison of
existing method lineages with reconstructable source provenance and sealed
certification.

We also note a naming coincidence. An unpublished repository,
`frankkramer-lab/AutoMIL`, provides a conventional MIL training and evaluation
pipeline for whole-slide images. It shares the name but not the scope of our
work, and to our knowledge it has no associated publication. Throughout this
paper, autoMIL refers to the autonomous experimentation framework introduced
here.

## BibTeX

```bibtex
@inproceedings{huang2024mlagentbench,
  title     = {{MLA}gent{B}ench: Evaluating Language Agents on Machine Learning
               Experimentation},
  author    = {Huang, Qian and Vora, Jian and Liang, Percy and Leskovec, Jure},
  booktitle = {Proceedings of the 41st International Conference on Machine Learning},
  volume    = {235},
  pages     = {20271--20309},
  year      = {2024}
}

@article{jiang2025aide,
  title   = {{AIDE}: {AI}-Driven Exploration in the Space of Code},
  author  = {Jiang, Zhengyao and Schmidt, Dominik and Srikanth, Dhruv and
             Xu, Dixing and Kaplan, Ian and Jacenko, Deniss and Wu, Yuxiang},
  journal = {arXiv preprint arXiv:2502.13138},
  year    = {2025}
}

@article{yamada2025aiscientistv2,
  title   = {The {AI} Scientist-v2: Workshop-Level Automated Scientific
             Discovery via Agentic Tree Search},
  author  = {Yamada, Yutaro and Lange, Robert Tjarko and Lu, Cong and Hu,
             Shengran and Lu, Chris and Foerster, Jakob and Clune, Jeff and
             Ha, David},
  journal = {arXiv preprint arXiv:2504.08066},
  year    = {2025}
}

@inproceedings{toledo2025aira,
  title     = {{AI} Research Agents for Machine Learning: Search, Exploration,
               and Generalization in {MLE}-bench},
  author    = {Toledo, Edan and others},
  booktitle = {Advances in Neural Information Processing Systems},
  volume    = {38},
  year      = {2025}
}

@article{hambardzumyan2026aira2,
  title   = {{AIRA}\_2: Overcoming Bottlenecks in {AI} Research Agents},
  author  = {Hambardzumyan, Karen and others},
  journal = {arXiv preprint arXiv:2603.26499},
  year    = {2026}
}

@article{wijk2024rebench,
  title   = {{RE-Bench}: Evaluating Frontier {AI} {R\&D} Capabilities of
             Language Model Agents Against Human Experts},
  author  = {Wijk, Hjalmar and others},
  journal = {arXiv preprint arXiv:2411.15114},
  year    = {2024}
}

@inproceedings{zhang2025mlrcbench,
  title     = {{MLRC-Bench}: Can Language Agents Solve Machine Learning
               Research Challenges?},
  author    = {Zhang, Yunxiang and Khalifa, Muhammad and Bhushan, Shitanshu and
               Murphy, Grant and Logeswaran, Lajanugen and Kim, Jaekyeom and
               Lee, Moontae and Lee, Honglak and Wang, Lu},
  booktitle = {Advances in Neural Information Processing Systems},
  volume    = {38},
  year      = {2025}
}

@article{liu2026amid,
  title   = {Towards Autonomous and Auditable Medical Imaging Model Development},
  author  = {Liu, Shengyuan and Jiang, Jia-Xuan and Zheng, Boyun and Wang,
             Cheng and Wang, Zipei and Pan, Wentao and Wu, Hongtao and Peng,
             Houwen and Gu, Yu and Sun, Lichao and Yuan, Yixuan},
  journal = {arXiv preprint arXiv:2607.10522},
  year    = {2026}
}

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

<!-- TODO(leo): expand the long AIRA/AIRA2/RE-Bench author lists in the final
     manuscript bibliography, confirm all preprints' latest publication status,
     and verify frankkramer-lab/AutoMIL is still unpublished at submission. -->
