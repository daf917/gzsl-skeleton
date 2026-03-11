# Configuration file for GZSL Skeleton Action Recognition

# Dataset configuration
dataset:
  name: "ntu60"  # nt60, nt120, ucf101, pku_mmd, hmdb51
  data_dir: "./data/ntu60/"
  num_classes: 60
  num_joints: 25
  split:
    seen: 55
    unseen: 5
  
# Model configuration
model:
  backbone: "shift-gcn"
  feature_dim: 256
  num_parts: 6  # Head, Torso, Left Arm, Right Arm, Left Leg, Right Leg
  temporal_window: 10
  dropout: 0.5
  
# Text encoder configuration
text_encoder:
  clip_model: "ViT-B/32"
  freeze_clip: True
  local_encoder_dim: 256
  
# Skeleton encoder configuration
skeleton_encoder:
  shift_gcn_pretrained: "./checkpoints/shift_gcn.pth"
  local_encoder_hidden_dim: 256
  
# Training configuration
training:
  batch_size: 32
  num_epochs: 100
  learning_rate: 0.0001
  weight_decay: 0.0001
  scheduler: "cosine"
  warmup_epochs: 5
  
  # Loss weights
  lambda_p: 1.0  # Primitive-level alignment
  lambda_g: 1.0  # Global-level alignment
  lambda_c: 0.5  # Consistency loss
  lambda_i: 0.3  # Independence loss
  
  # Temperature for contrastive loss
  temperature: 0.07
  
  # Device
  device: "cuda"
  num_workers: 4
  
# Prompt generation
prompt:
  llm_model: "gpt-3.5-turbo"
  template: "Describe the motion of {body_part} when performing the action '{action}' in a concise way."
  body_parts:
    - "head"
    - "torso" 
    - "left arm"
    - "right arm"
    - "left leg"
    - "right leg"

# Evaluation
evaluation:
  metrics: ["acc_s", "acc_u", "hm"]
  test_interval: 1
  
# Logging
logging:
  log_dir: "./logs/"
  checkpoint_dir: "./checkpoints/"
  save_interval: 10
