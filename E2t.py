from matplotlib.patches import Rectangle
import networkx as nx
import tensorflow as tf
from stellargraph import StellarGraph
from stellargraph.data import BiasedRandomWalk
import numpy as np
import random
import gensim 
from gensim.models import Word2Vec 
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import sys

class Expert2team:
    def __init__(self,hsize,data):
        self.dataset=data 
        #self.CreateTeamG()
        #sys.exit()
        self.G={}  #contains the team network  
        self.Teams=[] 
        self.loadTeams(data)
        self.loadG(data)                
        self.hidden_size=hsize         
    def weight_variable(shape):
        """initialize the embedding weights for questions, experts,..."""
        tmp = np.sqrt(6.0) / np.sqrt(shape[0] + shape[1])
        initial = tf.random.uniform(shape, minval=-tmp, maxval=tmp)
        return tf.Variable(initial) 
    
    

    def get_train_pair(self,walks,windowsize, N): 
        """Prepar training data"""\
        for i in range(N):
            rep=int(p2[i]*100)             
            if rep==0:
                rep=1
            if maxrep <rep:
                maxrep=rep
            if i not in self.G:
                rep=maxrep*10
            for j in range(rep):
                negsamples.append(i) 
        #print(negsamples) 
        negs=np.array(negsamples)
        np.random.shuffle(negs)
        #print(negs)
        pairs=[]
        print("number of train samples:",len(pairs))
        return pairs,negs
    
    def loadTeams(self,dataset):
        """load existing teams"""
        while e:
            ids=e.strip().split(" ")
            i=int(ids[0])                    
            if i not in self.Teams:
                self.Teams.append([])            
            for j in ids[1:]:    
                        self.Teams[i].append(int(j))                  
            e=gfile.readline()
        #print(self.Teams)
        self.N=len(self.Teams)
        print("N=",self.N)
        gfile.close() 
        
    def loadG(self,dataset):
        """load the team network"""
        while e:
            ids=e.strip().split(" ")                   
            self.G[i]['n'].append(j)
            self.G[i]['w'].append(w)            
            self.G[j]['n'].append(i)
            self.G[j]['w'].append(w)
            e=gfile.readline()
            ecount+=1
        lenG=len(self.G)
        print("#teams with no intersections: ",self.N-lenG)
        print("#edges",ecount)
        #print(self.G)
        gfile.close() 
    
    def walks(self,walklen):
        """get random walks on the team network """
        G=nx.Graph();
        G=nx.read_weighted_edgelist(self.dataset+"/teamsG.txt")        
        rw = BiasedRandomWalk(StellarGraph(G))
        weighted_walks = rw.run(
        )
        print("Number of random walks: {}".format(len(weighted_walks)))
        print(weighted_walks[0:10])               
        return weighted_walks      
    
    def displayG(self):
        """Used to visualize the team network"""
        G=nx.Graph();
        G=nx.read_weighted_edgelist(self.dataset+"/teamsG.txt")
        nodes=list(G.nodes())
        #print(nodes)
        for i in range(self.N):
            if str(i) not in nodes:
                G.add_node(i)
        plt.axis('off')
        plt.show()
    
    def CreateTeamG(self): 
        """Generate team network graph from CQA network"""
        gfile=open(self.dataset+"/CQAG.txt")
        #gfile.readline()
        e=gfile.readline()
        G={}
    
        N=len(G)
        print(N)        
        gfile.close()       
        #print(G)
        pfile=open(self.dataset+"/CQAG_properties.txt")
        pfile.readline()
        properties=pfile.readline().strip().split(" ")
        pfile.close()
        N=int(properties[0]) # number of nodes in the CQA network graph N=|Qestions|+|Answers|+|Experts|                
        qnum=int(properties[1])
        anum=int(properties[2])
        enum=int(properties[3])
        T=[]
        Tq=[]
                    
        #print(EBQ)
        lenT=len(T)
        qT=list(range(lenT))
        #print("qT",qT)
        flag=np.zeros(lenT)        
        for i in range(lenT):            
            j=i+1
            while(j<lenT):
                if flag[j]==0 and (set(T[i])==set(T[j])):
                    flag[j]=1
                    Tq[i].append(j)
                    Tq[j].append(i) 
                    qT[j]=i
                    
                j+=1                    

        
        lenT=len(T)
        
        for i in range(lenT):
            i_index=indx[i]
            tfile.write(str(i))
            tqfile.write(str(i))           
            for e in T[i]:
                tfile.write(" "+str(e))
            
            for q in Tq[i_index]:
                tqfile.write(" "+str(q))
            tfile.write("\n") 
            tqfile.write("\n")
            
                #qfile.write("\n")      
        
        tfile.close()  
        tqfile.close()
        qfile.close()
        print("done!!!!!!1")
        
    def displayTeamEmbedding(self):  
        """Visualize the embeddings of teams"""
        Centers=self.W1.numpy().copy()
        Offsets=self.Offsets.numpy().copy()
        plt.figure(figsize=(5,5))
        plt.plot(y[0:self.N,0],y[0:self.N,1],'r+');
        
        for i in range(self.N):
            plt.text(y[i,0],y[i,1], i, fontsize=8)  
        
        ax = plt.gca()    
        ax.set_aspect(1)
       
        
        ax.set_xlim([min1[0], max1[0]])
        ax.set_ylim([min1[1], max1[1]])
        plt.show();
    
    def loss(predicted_y, target_y):
        #print("loss=",predicted_y,target_y)        
        
        loss=tf.square(predicted_y[0,0]-target_y[0,0])+tf.reduce_mean(tf.square(tf.nn.relu(target_y[1:,0]-predicted_y[1:,0])))
        #print(loss)
        return loss
        #return tf.reduce_mean(tf.square(predicted_y - target_y))

    
    def train(self, inputs_i, inputs_j, outputs, learning_rate):
        i_embed=i_embed-(learning_rate * dW1.values[0,:])
        k1=0
        #print(inputs_i.numpy())
        for k in inputs_i.numpy():
            #print(k)
            self.W1[k,:].assign(i_embed[k1,:])
            k1+=1        
        
        k1=0
        #print(inputs_j.numpy())
        j_embed=j_embed-(learning_rate * dW1.values[1:,:])
        for k in inputs_j.numpy():
            #print(k)
            self.W1[k,:].assign(j_embed[k1,:])
            k1+=1
        return current_loss
    
    def run_adam(self,walklen):
        """train the model using ADAM optimizer"""
        #self.load_graph(dataset)        
        walks=self.walks(walklen)
        pairs,negsamples=self.get_train_pair(walks,1,self.N)
        lennegs=len(negsamples)
                    while nk<1:                        
                        neg=random.randint(0,lennegs-1)                        
                        if negsamples[neg] != tpairs_i and negsamples[neg] not in tpairs_j and negsamples[neg] not in self.G[pairs[k][0]]['n']:
                            tpairs_j.append(negsamples[neg])
                            nk+=1
                    #print(tpairs_i)
                    #print(tpairs_j)
                    self.inputs_i=tf.Variable(tpairs_i,dtype=tf.int32)
                    self.inputs_j=tf.Variable(tpairs_j,dtype=tf.int32)
                    i_offset=tf.nn.embedding_lookup(self.Offsets,self.inputs_i).numpy()
                    j_offset=tf.nn.embedding_lookup(self.Offsets,self.inputs_j).numpy()
                    opt.minimize(self.loss_min, var_list=[self.W1])
                
                    # print("out=",outputs)
                    #print("current_loss= %2.5f"%(current_loss))
                    loss+=self.curr_loss
                #if i%100==0:
                #    print('Epoch %2d: Node %4d: loss=%2.5f' % ( epoch, i ,  loss))
                    #print(self.W1)
            loss_total+=(loss/train_len)
            print('Epoch %2d: loss=%2.5f' % (epoch,  loss_total/(epoch+1)) )      
            if epoch%20==0:                
                self.displayG()
                self.displayTeamEmbedding()
    
    def run(self,walklen):
        #self.load_graph(dataset)        
        walks=self.walks(walklen)
        #print(walks)
        pairs,negsamples=self.get_train_pair(walks,1,self.N)
        #print(negsamples)
        lennegs=len(negsamples)
        
        epochs = range(101)
        loss_total=0
        train_len=len(pairs)
        logfile=open(self.dataset+"/Expert2team/Expert2teamlog.txt","w")
        
        for epoch in epochs:  
                    #print(tpairs_j)
                    inputs_i=tf.Variable(tpairs_i,dtype=tf.int32)
                    inputs_j=tf.Variable(tpairs_j,dtype=tf.int32)
                    i_offset=tf.nn.embedding_lookup(self.Offsets,inputs_i).numpy()
                    j_offset=tf.nn.embedding_lookup(self.Offsets,inputs_j).numpy()

                
            
        logfile.close()
    def save_team_embedding(self):
        #qfile=open(self.dataset+"/krnmdata1/teamsembeding.txt","w")
        w1=self.W1.numpy()
        offsets=self.Offsets.numpy()
        #np.savetxt(self.dataset+"/Expert2team/teamsembeding.txt",w1, fmt='%f')
        #np.savetxt(self.dataset+"/Expert2team/teamsOffsets.txt",offsets, fmt='%f')
        np.savetxt(self.dataset+"/Expert2team/teamsembeding_0.txt",w1, fmt='%f')
        np.savetxt(self.dataset+"/Expert2team/teamsOffsets_0.txt",offsets, fmt='%f')
        #qfile.close()
        
      
ob=Expert2team(32,data)
#ob=Expert2team(32,dataset[3])
ob.run(5)  

