import os
import pickle
file = open("example.pkl",'rb')
for i in range (500):
    data = pickle.load(file)
    path = str(i)+'.pkl'
    file1 = open(os.path.join('./example',path), 'wb')
    pickle.dump(data , file1)