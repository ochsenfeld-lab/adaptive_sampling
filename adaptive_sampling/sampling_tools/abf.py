from .enhanced_sampling import EnhancedSampling


class ABF(EnhancedSampling):
    def __init__(self, p1, p2, *args, **kwargs):
        super().__init__(*args, **kwargs)
