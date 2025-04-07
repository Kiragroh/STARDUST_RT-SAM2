import torch
import sys

def test_checkpoint(checkpoint_path):
    try:
        print(f"Attempting to load checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        print("Checkpoint loaded successfully!")
        if isinstance(checkpoint, dict):
            print(f"Checkpoint keys: {list(checkpoint.keys())}")
        else:
            print(f"Checkpoint type: {type(checkpoint)}")
        return True
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) > 1:
        checkpoint_path = sys.argv[1]
        test_checkpoint(checkpoint_path)
    else:
        print("Please provide a checkpoint path as an argument")
