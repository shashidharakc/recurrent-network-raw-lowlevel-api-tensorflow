# recurrent-network-raw-lowlevel-api-tensorflow


Purpose of this repo is to code RNN, LSTM and GRU, network using low level tensorflow API's, To get better understanding.

Implementations ar done based on the below equations : 

LSTM: 

    Forward Computation : 
    Ft = sigmoid(Whf*Ht^-1 + Wxf*Xt)
    It = sigmoid(Whi*Ht^-1 + Wxi*Xt)
    Ct_Dash = tanh(Whc*Ht^-1 + Wxc*Xt)
    Ct = Ft*Ct^-1 + It*Ct_Dash
    Ot = sigmoid(Who*Ht^-1 + Wxo*Xt)
    Ht = Ot*tanh(Ct)


GRU: 

    Forward Computation : 
    Zt = sigmoid(Whz*Ht^-1 + Wxz*Xt)
    Rt = sigmoid(Whr*Ht^-1 + Wxr*Xt)
    Ht_Dash = tanh((Rt*Whh*Ht^-1) + Wxh*Xt)
    Ht = (1-Zt)*Ht^-1+Zt*Ht_Dash
    
