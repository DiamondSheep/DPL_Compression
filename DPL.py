import torch

class DPL():
    def __init__(self, Data, DictSize=30, tau=0.05, gamma=0.0001):
        super().__init__()
        self.DictSize = DictSize
        self.tau = tau
        self.gamma = gamma
        self.DataMat = Data

        self.DictMat = torch.Tensor()
        self.CoefMat = torch.Tensor()
        #random initialization
        self.DictMat = self.normcol_equal(torch.rand(self.DataMat.shape[0], self.DictSize))
        self.P_Mat = self.normcol_equal(torch.rand(self.DataMat.shape[0], self.DictSize)).t()
        self.CoefMat = torch.zeros(self.DictSize, self.DataMat.shape[1])
        self.DataInvMat = torch.inverse(self.tau * torch.matmul(self.DataMat, self.DataMat.t()) + self.gamma * torch.eye(self.DataMat.shape[0]).cuda())
        if torch.cuda.is_available():
            self.DataMat = self.DataMat.cuda()
            self.DictMat = self.DictMat.cuda()
            self.P_Mat = self.P_Mat.cuda()
            self.CoefMat = self.CoefMat.cuda()
            self.DataInvMat = self.DataInvMat.cuda()

        self.UpdateA()
        #print("DPL Initialized.")
   
    def Update(self, iterations = 20, showFlag = True):
        for i in range(iterations):
            if showFlag:
                print("iteration: "+ str(i))
            self.UpdateP()
            self.UpdateD(iter_dict = 100)
            self.UpdateA()
        self.DictMat = self.DictMat.half()
        self.P_Mat = self.P_Mat.half()
        self.DataMat = self.DataMat.half()

    def UpdateA(self):
        I_Mat = torch.eye(self.DictSize)
        if torch.cuda.is_available():
            I_Mat = I_Mat.cuda()
        temp_1 = torch.inverse(torch.matmul(self.DictMat.t(), self.DictMat) + self.tau * I_Mat)
        temp_2 = torch.matmul(self.DictMat.t(), self.DataMat) + self.tau * torch.matmul(self.P_Mat, self.DataMat)
        self.CoefMat = torch.matmul(temp_1, temp_2)

    def UpdateP(self):

        self.P_Mat = self.tau * torch.matmul(torch.matmul(self.CoefMat, self.DataMat.t()), self.DataInvMat)

    def UpdateD(self, iter_dict = 100):
        #Dim = self.DataMat[0].shape[0]
        rate_rho = 1.2
        I_Mat = torch.eye(self.DictSize)
        TempT = torch.zeros(self.DictMat.shape)
        if torch.cuda.is_available():
            I_Mat = I_Mat.cuda()
            TempT = TempT.cuda()
        TempS = self.DictMat
        previousD = self.DictMat
        Iter = iter_dict
        ERROR = 1
        rho = 1
        while (Iter > 0 and (ERROR > 10 ** (-8))):
            temp_1 = (rho * (TempS - TempT) + torch.matmul(self.DataMat, self.CoefMat.t()))
            temp_2 = torch.inverse(rho * I_Mat + torch.matmul(self.CoefMat, self.CoefMat.t()))
            self.DictMat = torch.matmul(temp_1, temp_2)
            TempS = self.normcol_lessequal(self.DictMat + TempT)
            TempT = TempT + self.DictMat - TempS
            rho = rate_rho * rho
            ERROR = torch.mean((previousD - self.DictMat) ** 2)
            previousD = self.DictMat
            Iter -= 1

    def normcol_equal(self, matin):
        eps = 2.0 ** -52
        tempMat = torch.sqrt(torch.sum(matin * matin, axis=0) + eps)
        matout = matin / tempMat.repeat(matin.shape[0], 1)
        assert matout.shape == matin.shape
        return matout

    def normcol_lessequal(self, matin):
        eps = 2.0 ** -52
        tempMat = torch.sqrt(torch.sum(matin * matin, axis=0) + eps)
        tempOne = torch.ones(tempMat.shape)
        if torch.cuda.is_available():
            tempOne = tempOne.cuda()
        matout = matin / (torch.max(tempOne, tempMat).repeat(matin.shape[0], 1))
        assert matout.shape == matin.shape
        return matout

    def evaluate(self):
        err = torch.norm(torch.matmul(self.DictMat, torch.matmul(self.P_Mat, self.DataMat)) - self.DataMat)
        return err

if __name__ == "__main__":
    pass
    
