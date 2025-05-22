import torch
import torch.nn as nn
import os
import glob
import re
from datetime import datetime

class BaseModel(nn.Module):
    """
    Base model class supporting save/load/transfer of model weights and training epoch tracking.
    """

    def __init__(self):
        super(BaseModel, self).__init__()
        self.model_name = 'model'
        self.last_epoch = 0

    def load(self, model_dir='./checkpoints', mode=0, specified_path=None, optimizer=None):
        """
        Load model checkpoint.
        mode:
            0 - load the latest checkpoint by epoch
            1 - load {model_name}.pth
            2 - load latest modified checkpoint file in model_dir
        """
        load_path = self._get_load_path(specified_path, self.model_name, model_dir, mode)

        if load_path and os.path.exists(load_path):

            checkpoint = torch.load(load_path, weights_only=True)
            model_state_dict = checkpoint.get('model_state_dict')
            if model_state_dict:
                self.load_state_dict(model_state_dict)
                print(f"Model loaded from {load_path}, epoch: {checkpoint.get('epoch')}.")
            else:
                print(f"Failed to load model_state_dict from {load_path}.")
            self.last_epoch = checkpoint.get('epoch', 0)
        else:
            print(f"No model found for {self.model_name} in {model_dir}, starting from scratch.")
            self.last_epoch = 0
        return 0

    def transfer(self, path, strict=False):
        """
        Load compatible parameters from another checkpoint file. Shape-mismatched parameters will be skipped.
        """
        if os.path.exists(path):
            checkpoint = torch.load(path)
            checkpoint_state_dict = checkpoint['model_state_dict']
            model_state_dict = self.state_dict()
            new_state_dict = {}
            missing_parameters = []
            extra_parameters = []

            for name, parameter in model_state_dict.items():
                if name in checkpoint_state_dict:
                    if checkpoint_state_dict[name].size() == parameter.size():
                        new_state_dict[name] = checkpoint_state_dict[name]
                    else:
                        extra_parameters.append(name)
                else:
                    missing_parameters.append(name)

            self.load_state_dict(new_state_dict, strict=False)
            print(f"Model parameters transferred from {path}. Successfully loaded parameters: {len(new_state_dict)}")

            if missing_parameters:
                print(f"Parameters not found in the checkpoint and using default: {missing_parameters}")
            if extra_parameters:
                print(f"Parameters in checkpoint but not used due to size mismatch: {extra_parameters}")
        else:
            print(f"No checkpoint found at {path} to transfer parameters from.")

    def save(self, epoch, optimizer=None, model_dir='./checkpoints', mode=0):
        """
        Save model checkpoint.
        mode:
            0 - {model_name}_epoch_{epoch}.pth
            1 - {model_name}.pth
            2 - {model_name}_{timestamp}.pth
        """
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        self.last_epoch = epoch
        save_path = self._determine_save_path(model_dir, self.model_name, self.last_epoch, mode)
        save_dict = {'epoch': self.last_epoch, 'model_state_dict': self.state_dict()}
        if optimizer is not None:
            save_dict['optimizer_state_dict'] = optimizer.state_dict()
        torch.save(save_dict, save_path)
        print(f'Model saved to {save_path}')

    @staticmethod
    def _remove_batchnorm_state(checkpoint_model_state_dict):
        """Remove batch normalization layer's runtime state parameters."""
        return {
            k: v for k, v in checkpoint_model_state_dict.items()
            if "running_mean" not in k and "running_var" not in k and "num_batches_tracked" not in k
        }

    @staticmethod
    def _get_load_path(specified_path, model_name, model_dir, mode):
        if specified_path:
            return specified_path
        if mode == 0:
            pattern = os.path.join(model_dir, f'{model_name}_epoch_*.pth')
            files = glob.glob(pattern)
            epochs = [int(re.search('epoch_([0-9]+)', f).group(1)) for f in files if re.search('epoch_([0-9]+)', f)]
            if epochs:
                return os.path.join(model_dir, f'{model_name}_epoch_{max(epochs)}.pth')
        elif mode == 1:
            return os.path.join(model_dir, f'{model_name}.pth')
        elif mode == 2:
            files = glob.glob(os.path.join(model_dir, f'{model_name}_*.pth'))
            if files:
                return sorted(files, key=os.path.getmtime)[-1]
        return None

    @staticmethod
    def _determine_save_path(model_dir, model_name, last_epoch, mode):
        if mode == 0:
            return os.path.join(model_dir, f'{model_name}_epoch_{last_epoch}.pth')
        elif mode == 1:
            return os.path.join(model_dir, f'{model_name}.pth')
        else:
            timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            return os.path.join(model_dir, f'{model_name}_{timestamp}.pth')

