# Methods draft: sentence-level structural provenance

The English Methods draft is in [`methods_draft.tex`](methods_draft.tex). Sentence IDs below number its prose sentences in reading order, starting from the first sentence of the Problem Formulation subsection; equations and headings are not counted. The entries identify the structural template for each sentence; they do not imply that the cited papers contain PSPFlow's technical contributions. The draft text is newly written and technically grounded in the v4 implementation.

Primary sources:

- **DSN** — Bousmalis et al., [Domain Separation Networks, NeurIPS 2016](https://proceedings.neurips.cc/paper_files/paper/2016/file/45fbc6d3e05ebd93369ce542e8f2322d-Paper.pdf).
- **FM** — Lipman et al., [Flow Matching for Generative Modeling, ICLR 2023](https://arxiv.org/abs/2210.02747).
- **SimCLR** — Chen et al., [A Simple Framework for Contrastive Learning of Visual Representations, ICML 2020](https://proceedings.mlr.press/v119/chen20j.html).
- **Neural ODE** — Chen et al., [Neural Ordinary Differential Equations, NeurIPS 2018](https://proceedings.neurips.cc/paper/2018/hash/69386f6bb1dfed68692a24c8686939b9-Abstract.html).

| Draft sentence | Source sentence / location and structural role |
|---|---|
| M01 | DSN, Sec. 3 opening: establish the task setting before stating the model objective. |
| M02 | FM, Sec. 3.2 definitions preceding the conditional-flow objective: introduce variables and conditioning context before the equations. |
| M03 | DSN, Sec. 3 architecture overview: summarize the encoder, representation components, and decoder in computational order. |
| M04 | DSN, Sec. 3 task-loss setup: distinguish training supervision from the inputs used by the model. |
| M05 | SimCLR, Sec. 4 implementation details: state input duration/rate as a concise standalone setup sentence. |
| M06 | DSN, Sec. 3 shared encoder description: introduce a common encoder over the paired domains. |
| M07 | DSN, Sec. 3 architecture description: add the auxiliary representation path after the shared path. |
| M08 | DSN, Sec. 3 shared/private definitions: specify representation symbols, dimensions, and their role after introducing the modules. |
| M09 | DSN, Sec. 3.2 similarity-loss discussion: name the aligned features and then state their explicit objective. |
| M10 | SimCLR, Sec. 3 contrastive-loss construction: delimit the loss family and its ingredients; here the sentence clarifies that PSPFlow does not use InfoNCE. |
| M11 | DSN, Sec. 3 private encoder description: introduce the private branch and define its output separately. |
| M12 | DSN, Sec. 3 reconstruction objective: explain what the private component must preserve for reconstruction. |
| M13 | DSN, Eq. 5 and accompanying difference-loss paragraph: motivate a penalty that discourages shared/private redundancy. |
| M14 | DSN, Sec. 3 loss definitions: state the purpose of an additional regularizer independently. |
| M15 | FM, Sec. 3 conditional formulation: define the conditioning variable and the observations from which it is obtained. |
| M16 | FM, Sec. 3.2 conditional flow: state which context conditions the predictor before defining its architecture. |
| M17 | DSN, Sec. 3 architecture description: narrate the sequence of feature modulation, transformation, and output. |
| M18 | FM, Sec. 3.2 conditional-flow objective: introduce the regression target before writing the loss. |
| M19 | DSN, Sec. 3 decoder description: specify the representation inputs, fusion operation, and reconstruction path. |
| M20 | DSN, Sec. 3 reconstruction discussion: explain how output normalization supports the selected objective. |
| M21 | FM, Sec. 3.2 objective presentation: introduce a composite objective before expanding its terms. |
| M22 | FM, Sec. 3.2 conditional vector-field regression: define the waveform inputs and targets, then give the mathematical loss. |
| M23 | SimCLR, Sec. 3 loss construction: describe multiple objective components and their resolutions before specifying settings. |
| M24 | DSN, Sec. 3 reconstruction loss: introduce an auxiliary decoder and state its reconstruction target. |
| M25 | DSN, Sec. 3 loss aggregation: enumerate complementary terms and combine them into a weighted objective. |
| M26 | DSN, Sec. 3 learning regime: identify jointly trained components and fixed components. |
| M27 | FM, Sec. 3 conditional modeling setup: transition from the representation/context definition to the generative target. |
| M28 | FM, Sec. 3.2 conditional path: define the noise source, time variable, and interpolated state in sequence. |
| M29 | FM, Eq. 14 and Sec. 3.2: regress a learned vector field to the target path velocity. |
| M30 | FM, Sec. 3.2 path-estimation discussion: define an endpoint estimate and connect it to a task-level reconstruction loss. |
| M31 | Neural ODE, model description: explain inference as integrating a learned vector field from an initial state. |
| M32 | DSN, Sec. 4 experimental configuration: state which model variant produced the reported result and why. |
| M33 | DSN, Sec. 3 learning regime and Sec. 4 model selection: state the selection criterion and data boundary. |
| M34 | DSN, Sec. 3 task/data setup: define the data partition unit and the relationship among examples. |
| M35 | FM, Sec. 3 conditional sampling setup: separately describe how conditioning data are sourced at training and evaluation. |
| M36 | DSN, Sec. 4 experimental discussion: conclude the protocol description by explicitly delimiting supported generalization claims. |

## Implementation alignment notes

- The shared alignment term in v4 is normalized feature MSE, not an InfoNCE-style contrastive loss.
- The patient encoder consumes one same-subject reference PPG/ECG window and produces a 256-dimensional vector. The current data protocol does not enforce that this window predates the target, so “historical” or “longitudinal” should be used cautiously.
- The full-test result uses `stage_rep/best.pt` in deterministic `mean` mode. The flow checkpoint was evaluated during validation but was not used in the reported full test because its validation MAE/RMSE were substantially worse.
- The v4 benchmark is a within-subject split. It does not establish subject-disjoint generalization.
