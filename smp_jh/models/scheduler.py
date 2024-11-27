from torch.optim import lr_scheduler

class Scheduler:
    @staticmethod
    def get_scheduler(scheduler_type, optimizer, **kwargs):
        """
        Get the learning rate scheduler
        
        Args:
            scheduler_type (str): Type of scheduler
                - "step": StepLR
                - "cosine": CosineAnnealingLR
                - "reduce": ReduceLROnPlateau
            optimizer: The optimizer to schedule
            **kwargs: Additional arguments for the scheduler
        """
        if scheduler_type == "step":
            return lr_scheduler.StepLR(
                optimizer,
                step_size=kwargs.get('step_size', 10),
                gamma=kwargs.get('gamma', 0.1)
            )
        elif scheduler_type == "cosine":
            return lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=55, T_mult=2, eta_min=5e-6)
        elif scheduler_type == "reduce":
            return lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=kwargs.get('mode', 'max'),
                factor=kwargs.get('factor', 0.5),
                patience=kwargs.get('patience', 5),
                verbose=True
            )
        else:
            raise NotImplementedError(f"Scheduler type {scheduler_type} not implemented")