class BasecallerTester:
    
    def __init__(self, model, device, test_loader):
        self.model=model.to(device) 
        self.device=device
        self.test_loader=test_loader

    def __call__(self,):
        # TODO: implement tester, considering accuracy
        pass