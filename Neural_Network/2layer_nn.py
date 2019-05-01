import pandas as pd
import torch
import torch.nn.functional as F


if __name__=="__main__":
    hidden_1 = int(sys.argv[1])
    hidden_2 = int(sys.argv[2])
    lr = int(sys.argv[3])
    epoch = int(sys.argv[4])
    TrainSet = sys.argv[5]
    ValSet = sys.argv[6]

    with open(ValSet, 'r') as f:
        train_set = pd.read_csv(f)
    with open(TrainSet, 'r') as e:
        val_set = pd.read_csv(e)

    x_train, y_train = get_Data(train_set)
    x_valid, y_valid = get_Data(val_Set)

    model = TwoLayerNet(Xtrain.shape[1], hidden_1, hidden_2)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for i in range(epoch):
        print(i, ' ', end='')
        train(model, x_train, y_train, criterion, optimizer)

    valid(model, x_valid, y_valid, criterion)



def get_Data(dataset):
    X_train = standard(dataset)
    X_train = X_train.drop(['totals_transactionRevenue', 'fullVisitorId', 'visitStartTime'], axis=1)
    X_train = ordinary(X_train)

    Y_train = train_set[['totals_transactionRevenue']]
    Y_train = standardize(Y_train, Y_train)

    Xtrain = X_train.values
    X = torch.tensor(torch.from_numpy(Xtrain))
    Y = torch.tensor(torch.from_numpy(Y_train))

    return X,Y

def ordinary(data):
    categorical = ['channelGrouping','device_browser','device_operatingSystem','geoNetwork_country']
    boolean = ['trafficSource_adwordsClickInfo.isVideoAd','trafficSource_isTrueDirect']
    for col in categorical:
        attributes = data[col].unique()
        d = dict(enumerate(attributes,start=1))
        new = dict((v,k) for k,v in d.items())
        data[col].replace(new,inplace = True)
    for col in boolean:
        data[col].replace({True:1,False:0},inplace=True)
    return data

def standardize(TrainSet,TestSet):
    standard=np.array(TrainSet)
    standard_test=np.array(TestSet)
    mean_array=np.mean(standard,axis=0)
    stdev=np.std(standard,axis=0,ddof=1)
    if stdev==0:
        stdev=1
    outcome = (standard_test - mean_array) / stdev
    return outcome

def standard(data):
    numerical = ['totals_hits','totals_pageviews','totals_sessionQualityDim','totals_timeOnSite',
                 'totals_transactionRevenue','totals_transactions']
    for col in numerical:
        data[col]=standardize(data[col],data[col])
    return data

class TwoLayerNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(TwoLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        h_softmax = F.softmax(self.linear1(x.float()))
        y_pred = self.linear2(h_softmax)
        return y_pred

def train(model, x, y,criterion,optimizer):
    """
   train NN
    """
    model.train()
    y_pred = model(x)
    loss = criterion(y_pred.float(), y.float())
    print(loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


def valid(model, x_valid, y_valid,criterion):
    """
    test NN
    """
    model.eval()
    y_pred = model(x_valid)
    loss = criterion(y_pred.float(), y_valid.float())
    print(loss.item())
    return loss.item()

