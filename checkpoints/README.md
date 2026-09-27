# Checkpoints

Expected local layout:

```text
checkpoints/
├── teacher/
│   └── teacher_checkpoint.pt          # pretrained teacher model (it is also the prior model trained using ERA5 data)
└── progressive_4to1/
    └── student_round0_final.pt        # created by the distillation script
```

`distillation_config_4to1.yaml` uses
`teacher_checkpoint.pt` as its default teacher path. Change that field if the teacher checkpoint is stored elsewhere.
