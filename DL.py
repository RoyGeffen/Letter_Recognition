from tkinter import E
import numpy as np
import matplotlib.pyplot as plt
import os
import h5py
import math
from sklearn.metrics import classification_report, confusion_matrix




class DLModel:
    def __init__(self,name="Model"):
        self.name=name
        self.layers = [None]
        self.is_compiled=False
        self.prev_cost = 1E20
        self.num_minibatch_steps = 1


    def add(self,layer):
        self.layers.append(layer)
    

    def save_weights(self, path):
        for i in range(1, len(self.layers)):
            file_name = "Layer" + str(i)
            self.layers[i].save_weights(path, file_name)


    def _squared_means(self,AL,Y):
        return(np.square(Y-AL))


    def _squared_means_backwards(self,AL,Y):
        return(2*(AL-Y))


    def _cross_entropy(self,AL,Y):
        return(np.where(Y==0,-np.log(1-AL),-np.log(AL)))


    def _cross_entropy_backwards(self,AL,Y):
        dAL=np.where(Y==0,1/(1-AL),-1/AL)
        return(dAL)


    def _categorical_cross_entropy(self, AL, Y):
        errors = np.where(Y == 1, -np.log(AL), 0)
        return errors


    def _categorical_cross_entropy_backward(self, AL, Y):
        return AL - Y


    def compile(self,loss,threshold=0.5):
        self.loss=loss
        if(loss=="squared_means"):
            self.loss_forward=self._squared_means
            self.loss_backward=self._squared_means_backwards
        elif(loss=="cross_entropy"):
            self.loss_forward=self._cross_entropy
            self.loss_backward=self._cross_entropy_backwards
        elif(loss=="categorical_cross_entropy"):
            self.loss_forward=self._categorical_cross_entropy
            self.loss_backward=self._categorical_cross_entropy_backward
        else:
            raise Exception("Invalid loss function")
        self.threshold=threshold
        self.is_compiled=True


    def compute_cost(self,AL,Y):
        m=AL.shape[1]
        costs = self.loss_forward(AL,Y)
        J = (1/m) * np.sum(costs)

        L2_regularization_cost = 0
        for i in range(1,len(self.layers)):
            l = self.layers[i]
            if l.regularization == "L2":
                L2_regularization_cost += (1/2*m)*l.L2_lambda*np.sum(np.square(l.W))

        return J + L2_regularization_cost

    
    def train(self, X, Y, num_epocs, mini_batch_size = 64):
        L = len(self.layers)
        costs = []
        seed = 10
        step = 0
        for i in range(num_epocs):
            seed = seed + 1
            minibatches = DLModel.random_mini_batches(X, Y, mini_batch_size, seed)
            num_iterations = len(minibatches)
            print_ind = max(num_epocs * num_iterations * self.num_minibatch_steps // 100, 1)
            # loop over mini batches
            for j in range(num_iterations):
                (minibatch_X, minibatch_Y) = minibatches[j]
                # checking for adaptive layers
                for l in range(1,L):
                    if self.layers[l]._optimization == 'adaptive' and ( i> 0 or j > 0):
                        self.layers[l]._adaptive_alpha_W = self.layers[l].dW * self.layers[l].alpha 
                        self.layers[l]._adaptive_alpha_b = self.layers[l].db * self.layers[l].alpha
                # loop over each mini batch
                for a in range(self.num_minibatch_steps):
                    # forward propagation
                    Al = minibatch_X
                    for l in range(1,L):
                        Al = self.layers[l].forward_propagation(Al,False)
                    #backward propagation
                    dAl = self.loss_backward(Al, minibatch_Y)
                    for l in reversed(range(1,L)):
                        dAl = self.layers[l].backward_propagation(dAl)
                        # update parameters
                        self.layers[l].update_parameters()
                    #record progress
                    step += 1
                    if step % print_ind == 0 or step == 1:
                        J = self.compute_cost(Al, minibatch_Y)
                        costs.append(J)
                        print("progress is  "+str(step//print_ind*100//100)+"%:",str(J))

        return costs

    def predict(self, X, is_soft_softmax = True):
        Al = X
        L = len(self.layers)
        for i in range(1,L):
            Al = self.layers[i].forward_propagation(Al,True)

        if True:#Al.shape[0] > 1: # softmax
            if is_soft_softmax: 
                predictions = np.where(Al==Al.max(axis=0),1,0)
            else:
                predictions = Al
            return predictions           
        else:
            return Al > self.threshold
        
            

    def predict_softmax(self, X_test, Y_test):
        return confusion_matrix(X_test, Y_test)
        
    def __str__(self):
        s = self.name + " description:\n\tnum_layers: " + str(len(self.layers)-1) +"\n"
        if self.is_compiled:
            s += "\tCompilation parameters:\n"
            s += "\t\tprediction threshold: " + str(self.threshold) +"\n"
            s += "\t\tloss function: " + self.loss + "\n\n"

        for i in range(1,len(self.layers)):
            s += "\tLayer " + str(i) + ":" + str(self.layers[i]) + "\n"
        return s

    
    @staticmethod
    def to_one_hot(num_categories, Y):
        m = Y.shape[0]
        Y = Y.reshape(1, m)
        Y_new = np.eye(num_categories)[Y.astype('int32')]
        Y_new = Y_new.T.reshape(num_categories, m)
        return Y_new

    def check_backward_propagation(self, X, Y, epsilon=1e-7):
        #doesnt work properly
        L = len(self.layers)
        Al = X
        for l in range(1,L):
            Al = self.layers[l].forward_propagation(Al,False) 

        dAl = self.loss_backward(Al, Y)

        for l in reversed(range(1,L)):
            dAl = self.layers[l].backward_propagation(dAl)
            self.layers[l].update_parameters()

        for l in range(1, L):
            params_vec = self.layers[l].parms_to_vec()
            grad_vec = self.layers[l].gradients_to_vec()

            n = params_vec.shape[0]
            approx = np.zeros(n)
            grad = grad_vec
            for i in range(n):
                self.layers[l].vec_to_parms(params_vec)
                dummy1 = np.array(params_vec, copy=True)
                dummy1[i] += epsilon
                f_plus = self.layers[l].activation_forward(dummy1)
                dummy2 = np.array(params_vec, copy=True)
                dummy2[i] -= epsilon
                f_minus = self.layers[l].activation_forward(dummy2)
                approx[i] = ((f_plus-f_minus)/(2*epsilon))[0]
            diff = np.linalg.norm(grad - approx) / (np.linalg.norm(grad) + np.linalg.norm(approx))
            if diff > epsilon:
                return False, diff, l
            self.layers[l].vec_to_parms(params_vec)
        return True, diff, L


    def confusion_matrix(self, X, Y):
        prediction = self.predict(X)
        prediction_index = np.argmax(prediction, axis=0)
        Y_index = np.argmax(Y, axis=0)
        right = np.sum(prediction_index == Y_index)
        print("accuracy: ",str(right/len(Y[0])))
        print(confusion_matrix(prediction_index, Y_index))
        return (right/len(Y[0]))


    @staticmethod
    def random_mini_batches(X, Y, mini_batch_size = 64, seed = None):      
        if seed != None:
            np.random.seed(seed)

        m = X.shape[1]  # number of training examples

        mini_batches = []

        permutation = list(np.random.permutation(m))
        shuffled_X = X[:, permutation]
        shuffled_Y = Y[:, permutation].reshape((-1,m))

        # Minus the end case.
        num_complete_minibatches = math.floor(m/mini_batch_size) 
        for k in range(0, num_complete_minibatches):
            
            mini_batch_X = shuffled_X[:, mini_batch_size*k : (k+1) * mini_batch_size]
            mini_batch_Y = shuffled_Y[:, mini_batch_size*k : (k+1) * mini_batch_size]
            
            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)
    
        # Handling the end case (last mini-batch < mini_batch_size)
        if m % mini_batch_size != 0:

            mini_batch_X = shuffled_X[:, mini_batch_size*num_complete_minibatches : m]
            mini_batch_Y = shuffled_Y[:, mini_batch_size*num_complete_minibatches : m]

            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)
    
        return mini_batches





class DLLayer:
    def __init__(self,name,num_units,input_shape, activation="relu" 
    ,W_initialization="random",learning_rate=0.01,optimization=None, 
    random_scale=0.01,leaky_relu_d=0.01,adaptive_cont=1.1, 
    adaptive_switch=0.5,activation_trim=1e-10 , regularization = None):

        self.name=name
        self._num_units=num_units
        self._input_shape=input_shape
        self._activation=activation
        self.random_scale=random_scale
        self.alpha = learning_rate
        self._optimization = optimization
        self.activation_trim = activation_trim
         
        if self._activation=="leaky_relu":
            self.leaky_relu_d=leaky_relu_d

        self.init_functions()
        self.init_weights(W_initialization)
        self.init_regularization(regularization)
        self.init_optimization(optimization,adaptive_cont,adaptive_switch)


    def init_weights(self, W_initialization):
        self.b = np.zeros((self._num_units,1), dtype=float)
        if W_initialization=="random":
            self.W = np.random.randn(self._num_units, *(self._input_shape)) * self.random_scale
        elif W_initialization=="zeros":
            self.W=np.zeros((self._num_units,*self._input_shape),dtype=float)
        elif W_initialization=="He":
            self.W = np.random.randn(self._num_units, *(self._input_shape))*np.sqrt(1/sum(self._input_shape))
        elif W_initialization=="Xaviar":
            self.W = np.random.randn(self._num_units, *(self._input_shape))*np.sqrt(2/sum(self._input_shape))
        else:
            try:
                with h5py.File(W_initialization, 'r') as hf:
                    self.W = hf['W'][:]
                    self.b = hf['b'][:]
            except (FileNotFoundError):
                raise NotImplementedError("Unrecognized initialization:", W_initialization)        
    

    def init_functions(self):
        functions = {"sigmoid":self._sigmoid,"tanh":self._tanh,"relu":self._relu, \
                    "leaky_relu":self._leaky_relu,"trim_sigmoid":self._trim_sigmoid,"trim_tanh":self._trim_tanh, \
                    "softmax": self._softmax, "trim_softmax": self._trim_softmax}
        backwards_functions = {"sigmoid":self._sigmoid_backward,"tanh":self._tanh_backward,"relu":self._relu_backward, \
                    "leaky_relu":self._leaky_relu_backward,"trim_sigmoid":self._trim_sigmoid_backward, \
                    "trim_tanh":self._trim_tanh_backward, "softmax": self._softmax_backward, "trim_softmax": self._softmax_backward}
        try:
            self.activation_forward=functions[self._activation]
            self.activation_backward = backwards_functions[self._activation]
        except:
            raise Exception("activation function is not one of :"+ str(([key for key in functions])).strip("[]"))
        

                                                                    #0.6
    def init_regularization(self, regularization, dropout_keep_prob = 0.01, L2_lambda = 0.1):
        self.regularization = regularization
        if regularization == "dropout":
            self.dropout_keep_prob = dropout_keep_prob
        elif regularization == "L2":
            self.L2_lambda = L2_lambda
        elif regularization != None:
            raise Exception("invalid regularization")
        

    def init_optimization(self,optimization,adaptive_cont=1.1,adaptive_switch=0.5):
        if self._optimization=="adaptive":
            self._adaptive_alpha_b = np.full((self._num_units, 1), self.alpha)
            self._adaptive_alpha_W = np.full((self._num_units, *(self._input_shape)), self.alpha)
            self.adaptive_cont=adaptive_cont
            self.adaptive_switch = adaptive_switch

        elif self._optimization == "momentum":
            self.momentum_beta = 0.9
            self._momentum_v_dW = np.zeros(self.W.shape, dtype=float)
            self._momentum_v_db = np.zeros(self.b.shape, dtype=float)
        
        elif self._optimization == "adam":
            self._adam_beta1 = 0.9
            self._adam_beta2 = 0.999
            self._adam_epsilon = 1e-8
            self._adam_v_dW = np.zeros(self.W.shape, dtype=float)
            self._adam_v_db = np.zeros(self.b.shape, dtype=float)
            self._adam_s_dW = np.zeros(self.W.shape, dtype=float)
            self._adam_s_db = np.zeros(self.b.shape, dtype=float)
        

    def forward_propagation(self,A_prev,is_predict):
        if not is_predict:
            if self.regularization == "dropout":
                self._D = np.random.rand(A_prev.shape[0], 1)
                self._D = np.where(self._D < self.dropout_keep_prob, 0, 1)
                A_prev = np.multiply(A_prev, self._D)
                A_prev = A_prev / self.dropout_keep_prob

        self._A_prev= np.array(A_prev,copy=True)
        self._Z = np.dot(self.W, A_prev) + self.b
        A = self.activation_forward(self._Z)
        return(A)


    def backward_propagation(self, dA):
        m = self._A_prev.shape[1]
        dZ = self.activation_backward(dA)
        dA_Prev = np.dot(self.W.T ,dZ)

        self.db = (1.0/m)*(np.sum(dZ, axis=1 ,keepdims=True))
        self.dW = (1.0/m)*(np.dot(dZ,self._A_prev.T))

        if self.regularization == "dropout":
            dA_Prev *= self._D
            dA_Prev /= self.dropout_keep_prob
        elif self.regularization == "L2":
            self.dW += (self.L2_lambda / m) * self.W

        return dA_Prev
    

    def save_weights(self, path,file_name):
        if not os.path.exists(path):
            os.makedirs(path)

        with h5py.File(path+"/"+file_name+'.h5', 'w') as hf:
            hf.create_dataset("W",  data=self.W)
            hf.create_dataset("b",  data=self.b)


    def _softmax(self, Z):
        eZ = np.exp(Z)
        return eZ / np.sum(eZ, axis=0)
    def _sigmoid(self,Z):
        return(1/(1+np.exp(-Z)))
    def _tanh(self,Z):
        return(np.tanh(Z))
    def _relu(self,Z):
        return(np.where(Z>0,Z,0))
    def _leaky_relu(self,Z):
        return(np.where(Z>0,Z,Z*self.leaky_relu_d))
    def _trim_sigmoid(self,Z):
        with np.errstate(over='raise', divide='raise'):
            try:
                A = 1/(1+np.exp(-Z))
            except FloatingPointError:
                Z = np.where(Z < -100, -100,Z)
                A = 1/(1+np.exp(-Z))
        TRIM = self.activation_trim
        if TRIM > 0:
            A = np.where(A < TRIM,TRIM,A)
            A = np.where(A > 1-TRIM,1-TRIM, A)
        return A
    def _trim_tanh(self,Z):
        A = np.tanh(Z)
        TRIM = self.activation_trim
        if TRIM > 0:
            A = np.where(A < -1+TRIM,TRIM,A)
            A = np.where(A > 1-TRIM,1-TRIM, A)
        return A
    def _trim_softmax(self, Z):
        Z = np.where(Z > 100, 100,Z)
        Z = np.where(Z < -100, -100,Z)
        eZ = np.exp(Z)
        A = eZ/np.sum(eZ, axis=0)
        return A


    def _sigmoid_backward(self,dA):
        A = self._sigmoid(self._Z)
        dZ = dA * A * (1-A)
        return dZ
    def _tanh_backward(self,dA):
        A = self._tanh(self._Z)
        dZ = dA *(1-np.multiply(A,A))
        return dZ
    def _relu_backward(self,dA):
        dZ = np.where(self._Z<0,0,dA)
        return(dZ)
    def _leaky_relu_backward(self,dA):
        dZ = np.where(self._Z<0,dA*self.leaky_relu_d,dA)
        return(dZ)    
    def _trim_sigmoid_backward(self,dA):
            A = self._trim_sigmoid(self._Z)
            dZ = dA * A * (1-A)
            return dZ
    def _trim_tanh_backward(self,dA):
            A = self._trim_tanh(self._Z)
            dZ = dA * (1-A**2)
            return dZ
    def _softmax_backward(self, dA):
        return dA


    def __str__(self):
        s = self.name + " Layer:\n"
        s += "\tnum_units: " + str(self._num_units) + "\n"
        s += "\tactivation: " + self._activation + "\n"
        if self._activation == "leaky_relu":
            s += "\t\tleaky relu parameters:\n"
            s += "\t\t\tleaky_relu_d: " + str(self.leaky_relu_d)+"\n"
        s += "\tinput_shape: " + str(self._input_shape) + "\n"
        s += "\tlearning_rate (alpha): " + str(self.alpha) + "\n"
        #optimization
        if self._optimization == "adaptive":
            s += "\t\tadaptive parameters:\n"
            s += "\t\t\tcont: " + str(self.adaptive_cont)+"\n"
            s += "\t\t\tswitch: " + str(self.adaptive_switch)+"\n"
        if self.regularization != None:
            if self.regularization == "L2":
                s += "\t\tregularization: L2\n"
                s += "\t\t\tL2 parameters:\n"
                s += "\t\t\t\tlambda: " + str(self.L2_lambda)+"\n"
            elif self.regularization == "dropout":
                s += "\t\tregularization: dropout\n"
                s += "\t\t\tdropout parameters:\n"
                s += "\t\t\t\tkeep prob: " + str(self.dropout_keep_prob)+"\n"

        # parameters
        s += "\tparameters:\n\t\tb.T: " + str(self.b.T) + "\n"
        s += "\t\tshape weights: " + str(self.W.shape)+"\n"
        plt.hist(self.W.reshape(-1))
        plt.title("W histogram")
        plt.show()
        return s


    def update_parameters(self,  t=1):
        if self._optimization==None:
            self.W-=self.dW*self.alpha
            self.b-=self.db*self.alpha
        elif self._optimization=="adaptive":
            self._adaptive_alpha_W = np.where(self._adaptive_alpha_W*self.dW>0, \
            self._adaptive_alpha_W*self.adaptive_cont,self._adaptive_alpha_W*(-self.adaptive_switch))

            self._adaptive_alpha_b = np.where(self._adaptive_alpha_b*self.db>0, \
            self._adaptive_alpha_b*self.adaptive_cont,self._adaptive_alpha_b*(-self.adaptive_switch))

            self.W-=self._adaptive_alpha_W
            self.b-=self._adaptive_alpha_b

        elif self._optimization == "momentum":
            #update v_db and v_dW
            self._momentum_v_dW = self.momentum_beta * self._momentum_v_dW + (1-self.momentum_beta) * self.dW
            self._momentum_v_db = self.momentum_beta * self._momentum_v_db + (1-self.momentum_beta) * self.db
            #update b and W
            self.W = self.W - self.alpha * self._momentum_v_dW
            self.b = self.b - self.alpha * self._momentum_v_db
        
        elif self._optimization == "adam":
            #update v
            self._adam_v_dW = self._adam_beta1 * self._adam_v_dW + (1 - self._adam_beta1) * self.dW
            self._adam_v_db = self._adam_beta1 * self._adam_v_db + (1 - self._adam_beta1) * self.db
            #calculate vHat 
            self._adam_vHat_dW = self._adam_v_dW / (1 - self._adam_beta1**t)
            self._adam_vHat_db = self._adam_v_db / (1 - self._adam_beta1**t)
            #update s
            self._adam_s_dW = self._adam_beta2 * self._adam_s_dW + (1 - self._adam_beta2) * (self.dW**2)
            self._adam_s_db = self._adam_beta2 * self._adam_s_db + (1 - self._adam_beta2) * (self.db**2)
            #calculate sHat
            self._adam_sHat_dW = self._adam_s_dW / (1 - self._adam_beta2**t)
            self._adam_sHat_db = self._adam_s_db / (1 - self._adam_beta2**t)
            #update W and b
            self.W = self.W - self.alpha * self._adam_vHat_dW / (np.sqrt(self._adam_sHat_dW+self._adam_epsilon))
            self.b = self.b - self.alpha * self._adam_vHat_db / (np.sqrt(self._adam_sHat_db+self._adam_epsilon))



    def parms_to_vec(self):
        return np.concatenate((np.reshape(self.W,(-1,)), np.reshape(self.b, (-1,))), axis=0)

    def vec_to_parms(self, vec):
        self.W = vec[0:self.W.size].reshape(self.W.shape)
        self.b = vec[self.W.size:].reshape(self.b.shape)

    def gradients_to_vec(self):
        return np.concatenate((np.reshape(self.dW,(-1,)), np.reshape(self.db, (-1,))), axis=0)