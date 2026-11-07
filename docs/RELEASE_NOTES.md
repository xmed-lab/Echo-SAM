# Source release notes

## Included material

The supplied manuscript's `code/` directory is preserved as the importable
`sam3/` package: 197 source, configuration, vocabulary, prompt, and test-asset
files. Python bytecode caches are excluded. The release also contains five
manuscript figures, PNG previews, installation metadata, and documentation.
The original manuscript directory is not modified.

The release retains the supplied backbone, image/video models, training and
evaluation infrastructure, few-shot/LoRA components, agent utilities, and
cardiac extensions. Optional upstream agent/video benchmark workflows can
require additional dependencies beyond the cardiac `train` installation.

## Portability changes

- Cardiac configurations use `ECHO_SAM_DATA_ROOT`, `SAM3_CHECKPOINT`, and optional
  `ECHO_SAM_OUTPUT_DIR` environment variables, with repository-relative graph
  and tokenizer paths. The `cpus` device spelling is normalized to `cpu`.
- The training launcher respects the caller's `CUDA_VISIBLE_DEVICES`.
- The generic evaluation configuration references its existing validation
  normalization values and uses gradient accumulation of 1 instead of missing
  interpolation keys. Its undefined training-loss reference is replaced with
  the existing `DummyLoss`, as appropriate for evaluation.
- Medical loss configuration keys are aligned with `MedicalAuxLoss`:
  `use_pseudo_label_loss` becomes `use_pseudo_labels`; the redundant
  `use_pseudo_mask_loss` flag is removed because the implementation handles
  mask and box losses together; `pseudo_mask_weight` becomes `loss_mask_weight`
  with its supplied numeric value retained.
- Zero-shot datasets pass an empty support-dataset list, avoiding Hydra's
  otherwise eager loading of unused few-shot annotations.
- The zero-shot evaluation configuration sets model evaluation mode and
  requires `ECHO_SAM_CHECKPOINT` for the trainer checkpoint. The model builder
  first loads `SAM3_CHECKPOINT`; the evaluator then restores the complete
  trainer state dictionary from `ECHO_SAM_CHECKPOINT`.

These are release compatibility changes. The model, graph, geometric rules,
loss implementation, and metric algorithms have not been rewritten.

## Reproduction scope

This snapshot does not contain the raw-dataset converters, original split
manifests, trained Echo-SAM weights, or launchers for all ablations, shared-box
experiments, anchor robustness experiments, and ASD evaluation.

The provided graph fusion uses averaged neighbor embeddings and momentum
buffers. Geometric reasoning is implemented with explicit cardiac box/mask
rules and sampled geometric perturbations. A separate implementation of the
manuscript's learned relation-relevance scores and posterior candidate scoring
was not found in the supplied files. The image processor and dataset evaluator
do not connect predicted seen anchors, geometric target support, and final
unseen-target decoding into the complete inference procedure described in the
manuscript. The geometry utilities are used by the medical pseudo-label loss
and the annotation-based diagnostic visualizer.

The graph JSON contains 15 named anatomical structures plus `other`.
`MedicalKnowledgeBase` deduplicates adjacency pairs into undirected edges;
counts in a directed adjacency list and the stored edge embedding table differ.

The supplied validation paths and support-based development configurations do
not themselves enforce the paper's zero-shot protocol. Training and checkpoint
selection must use only seen-category annotations (LA, RA, RV). LV and Myo
labels belong only in offline evaluation. Confirm dataset identities and
splits before interpreting results as a reproduction of manuscript tables.

## Evaluation details

`sam3/eval/coco_eval_offline.py` selects the highest-IoU prediction for each
reference instance. Missing predictions receive zero Dice/IoU and infinite
boundary distances at the instance level. The current aggregation then
excludes infinite boundary distances when computing HD95/ASSD means. Thus a
finite aggregate does not imply that all targets were detected. This behavior
is retained from the supplied code and should be distinguished from averaging
all instance errors, as described in the manuscript.

The evaluator also initializes from the base model before restoring the
research checkpoint. Passing an Echo-SAM trainer checkpoint only as the base
`SAM3_CHECKPOINT` is insufficient: the model builder's loader expects the
upstream `detector.*` key layout. Use the documented `ECHO_SAM_CHECKPOINT` route
for the complete research model state.

## Verification

The release is checked with Python compilation, package/resource checks,
Hydra configuration composition, medical-loss construction, and focused CPU
smoke tests. Full training and dataset evaluation require external data and
checkpoints and have not been run for this release. Manuscript results in the
README are reported values, not measurements from these checks.
