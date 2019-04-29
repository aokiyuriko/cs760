import pandas as pd
import torch


def neuralnet(data):

    ## initialize tensor for inputs, and outputs
    Y = data[['totals_transactionRevenue']]
    X = data.drop(['totals_transactionRevenue','fullVisitorId','visitStartTime'], axis=1)

    ##parameters
    Max_Epoch = 50
    n_input, n_hidden, n_output = X.shape[1], 8, 1

    Y = torch.tensor(Y.values)
    Y = Y.float()
    X = torch.tensor(X.values)
    X = X.float()
    net = torch.nn.Sequential(torch.nn.Linear(n_input, n_hidden),
                              torch.nn.Sigmoid(),
                              torch.nn.Linear(n_hidden, n_output),
                              torch.nn.Sigmoid())
    loss_function = torch.nn.MSELoss()

    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

    for epoch in range(Max_Epoch):
        # Forward Propagation
        y_pred = net(X)
        # Compute and print loss
        loss = loss_function(y_pred, Y)
        print('epoch: ', epoch, ' loss: ', loss.item())
        # Zero the gradients
        optimizer.zero_grad()

        # perform a backward pass (backpropagation)
        loss.backward()

        # Update the parameters
        optimizer.step()




## one hot encoding
def one_hot(data):
    categorical = ['channelGrouping', 'device_browser', 'device_operatingSystem', 'geoNetwork_country']
    boolean = ['trafficSource_adwordsClickInfo.isVideoAd', 'trafficSource_isTrueDirect']
    for col in categorical:
        new = pd.get_dummies(data[col])
        new.columns = [f"{col}.{subcolumn}" for subcolumn in new.columns]
        data = data.drop(col, axis=1)
        data = data.merge(new, right_index=True, left_index=True)
    for col in boolean:
        data[col].replace({True: 1, False: 0}, inplace=True)

    return data


## ordianal encoding
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

def main():
    train = pd.read_pickle('train_set_0.pkl')
    # train = train.iloc[0:1000,:]
    test = pd.read_pickle('test_set.pkl')

    train_encoded = ordinary(train)
    test_encoded  = ordinary(test)

    neuralnet(train_encoded)

if __name__=="__main__":
    main()




