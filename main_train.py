''' training script of DECA
'''
import os, sys
import numpy as np
import yaml
import torch
import torch.backends.cudnn as cudnn
import torch
import shutil
from copy import deepcopy

torch.cuda.empty_cache()

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
np.random.seed(0)

def main(cfg):
    print("\n" + "="*50)
    print("STARTING MAIN TRAINING PROCESS")
    print("="*50)
    # creat folders
    try:
        # Create output directories
        print("\n[DEBUG] Creating output directories...")
        log_dir = os.path.join(cfg.output_dir, cfg.train.log_dir)
        vis_dir = os.path.join(cfg.output_dir, cfg.train.vis_dir)
        val_vis_dir = os.path.join(cfg.output_dir, cfg.train.val_vis_dir)
        
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(vis_dir, exist_ok=True)
        os.makedirs(val_vis_dir, exist_ok=True) 
        # os.makedirs(os.path.join(cfg.output_dir, cfg.train.log_dir), exist_ok=True)
        # os.makedirs(os.path.join(cfg.output_dir, cfg.train.vis_dir), exist_ok=True)
        # os.makedirs(os.path.join(cfg.output_dir, cfg.train.val_vis_dir), exist_ok=True)

        print(f"[DEBUG] Created directories:")
        print(f"  - Logs: {log_dir}")
        print(f"  - Visualizations: {vis_dir}")
        print(f"  - Validation: {val_vis_dir}")

        # Save full configuration
        config_path = os.path.join(log_dir, 'full_config.yaml')
        print(f"\n[DEBUG] Saving full configuration to: {config_path}")
        with open(config_path, 'w') as f:
            yaml.dump(cfg, f, default_flow_style=False)

        # Copy original config file
        backup_path = os.path.join(cfg.output_dir, 'config.yaml')
        print(f"[DEBUG] Backing up config to: {backup_path}")
        shutil.copy(cfg.cfg_file, backup_path)

        # with open(os.path.join(cfg.output_dir, cfg.train.log_dir, 'full_config.yaml'), 'w') as f:
        #     yaml.dump(cfg, f, default_flow_style=False)
        # shutil.copy(cfg.cfg_file, os.path.join(cfg.output_dir, 'config.yaml'))
        
        # CUDA settings
        print("\n[DEBUG] Configuring CUDA settings...")
        # cudnn related setting
        cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.enabled = True
        print(f"  - CuDNN Benchmark: {cudnn.benchmark}")
        print(f"  - Deterministic: {torch.backends.cudnn.deterministic}")

        # start training
        # deca model
        from decalib.deca import DECA
        from decalib.trainer import Trainer

        print(f"[DEBUG] Model configuration:")
        print(f"  - Rasterizer type: pytorch3d")
        print(f"  - Model path: {getattr(cfg, 'model_path', 'NOT SPECIFIED')}")
        print(f"  - Data path: {getattr(cfg, 'data_path', 'NOT SPECIFIED')}")
        cfg.rasterizer_type = 'pytorch3d'
        deca = DECA(cfg)
        print("[DEBUG] DECA model initialized successfully")
        # Initialize trainer
        print("\n[DEBUG] Initializing Trainer...")
        trainer = Trainer(model=deca, config=cfg)
        print("[DEBUG] Trainer initialized successfully")

        ## start train
                # Start training
        print("\n" + "="*50)
        print("STARTING TRAINING PROCESS")
        print("="*50)
        trainer.fit()
        print("\n[DEBUG] Training completed successfully!")

    except Exception as e:
        print("\n[ERROR] Critical failure during execution:")
        print(f"Type: {type(e).__name__}")
        print(f"Message: {str(e)}")
        print("\n[DEBUG] Traceback:")
        import traceback
        traceback.print_exc()
        
        sys.exit(1)

if __name__ == '__main__':
    from decalib.utils.config import parse_args
    cfg = parse_args()
    if cfg.cfg_file is not None:
        exp_name = cfg.cfg_file.split('/')[-1].split('.')[0]
        cfg.exp_name = exp_name
    main(cfg)

# run:
# python main_train.py --cfg configs/release_version/deca_pretrain.yml 