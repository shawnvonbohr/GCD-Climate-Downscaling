# Checkpoints

Expected local layout:

```text
checkpoints/
├── teacher/
│   ├── teacher_checkpoint.pt          # final ERA5-trained diffusion prior / teacher model
│   └── teacher_checkpoint_epoch*.pt   # periodic snapshots written during prior training
└── progressive_4to1/
    └── student_round0_final.pt        # created by the distillation script
```

`prior_training_config.yaml` writes the final teacher model to
`teacher_checkpoint.pt`. `distillation_config_4to1.yaml` uses this file as its
default teacher path. Change the relevant configuration field if the checkpoint
is stored elsewhere.
