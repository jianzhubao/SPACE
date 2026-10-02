"""Run after installing space-calibration; replace the SFT/output paths."""
from space_calibration import space_calibrate

if __name__ == '__main__':
    output = space_calibrate(
        pre_sft='Qwen/Qwen2.5-7B',
        post_sft='./sft-checkpoint',
        output_dir='./space-checkpoint',
        rho=0.5,
        alpha=1.0,
        device='auto',
    )
    print(output)
