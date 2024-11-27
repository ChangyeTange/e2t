from stellargraph import StellarGraph 
from stellargraph.data import BiasedRandomWalk
from matplotlib.patches import Rectangle
import networkx as nx
import tensorflow as tf
import numpy as np 
import random
import gensim 
from gensim.models import Word2Vec 
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import sys
class ExpertsEmbeding:        
    def __init__(self,hsize,data):        
        self.dataset=data
        #3self.load_graph()
        #self.save_qraph()
        #sys.exit()
        pfile=open(self.dataset+"/CQAG_properties.txt")
        pfile.readline()
        properties=pfile.readline().strip().split(" ")
        pfile.close()
        self.N=int(properties[0]) # number of nodes in the CQA network graph N=|Qestions|+|Answers|+|Experts|                
        self.qnum=int(properties[1])
        self.anum=int(properties[2])
        self.enum=int(properties[3])
        self.G={}  
        self.loadG()               
        #self.displayG()
        
        # load team centers and offsets
        self.teamcenters=np.loadtxt(self.dataset+"/e2t/teamsembeding.txt")
        print(self.teamcenters)
        self.teamoffsets=np.loadtxt(self.dataset+"/e2t/teamsOffsets.txt")
        print(self.teamoffsets)
        
        
        # load expert best team
        gfile=open(self.dataset+"/ExpertBestQuetionAnswer.txt")        
        gfile.readline()
        line=gfile.readline().strip()
        self.ebt={}
        while line:
            ids=line.split(" ")
            self.ebt[int(ids[0])]=[int(ids[1]),int(ids[2]),float(ids[3])]
            line=gfile.readline()
        print("ebt=",self.ebt)
        gfile.close()
        
    
    def weight_variable(shape):
        tmp = np.sqrt(6.0) / np.sqrt(shape[0] + shape[1])
        initial = tf.random.uniform(shape, minval=-tmp, maxval=tmp)
        return tf.Variable(initial)
    
    def weight_variable_experts(self,shape): 
              
        x=[]
        for i in range(self.enum):
            expertid=i+self.qnum+self.anum
            eteamid=self.ebt[expertid][0]
            offsets= self.teamoffsets[eteamid]
            #print(offsets)
            r = offsets * np.sqrt(np.random.uniform(0,1))
            #print("r",r)
            theta = np.random.uniform(0,1) * 2 * 3.14
        initial=np.array(x,dtype=np.float32)        
        return tf.Variable(initial)
    
    def loadG(self):  
        """
        load CQA graph
        """    
        gfile=open(self.dataset+"/CQAG.txt")
        
        e=gfile.readline()
        self.G={}
        while e:
            ids=e.split(" ")
            i=int(ids[0])
            j=int(ids[1])
            w=int(ids[2])
            
            if i not in self.G:
                self.G[i]={'n':[],'w':[]}
            
            if j not in self.G:    
                        self.G[j]={'n':[],'w':[]}
                    
            self.G[i]['n'].append(j)
            self.G[i]['w'].append(w) 
            
            self.G[j]['n'].append(i)
            self.G[j]['w'].append(w)
            e=gfile.readline()
        self.N=len(self.G)
        #print(self.G)
        gfile.close()   
            
    def load_graph(self): 
        """generate graph CQA from data"""    
        self.G={}
        qpfile=open(self.dataset+"q_answer_ids_score.txt")
        qpfile.readline()
        line=qpfile.readline().strip()
        qids=[]
        aids=[]
        eids=[]
        while line:
            qp=line.split(" ")
            qid=int(qp[0].strip())            
            if qid not in qids:
                qids.append(qid)
            caids=qp[1::2] 
            for aid in caids:
                if int(aid) not in aids:
                    aids.append(int(aid))
            line=qpfile.readline().strip()    
        qpfile.close()  
        print(len(qids))
        print(len(aids))
        pufile=open(self.dataset+"postusers.txt")
        pufile.readline()
        line=pufile.readline().strip()
        while line:
            ids=line.split(" ")
            eid=int(ids[1].strip())            
            if eid not in eids:
                eids.append(eid)
            line=pufile.readline().strip() 
        pufile.close()
        print(len(eids))
        
        self.qnum, self.anum, self.enum=len(qids), len(aids), len(eids)
        self.N=len(qids)+len(aids)+len(eids)
        pufile=open(self.dataset+"/krnmdata1/postusers.txt")
        pufile.readline()
        line=pufile.readline().strip()
        while line:
            ids=line.split(" ")
            aid=aids.index(int(ids[0].strip()))+len(qids)
            eid=eids.index(int(ids[1].strip()))+len(qids)+len(aids)           
                      
            if eid not in self.G:
                self.G[eid]={'n':[aid],'w':[self.G[aid]['w'][0]]}
                
            else:
                self.G[eid]['n'].append(aid)
                self.G[eid]['w'].append(self.G[aid]['w'][0])
            self.G[aid]['n'].append(eid)
            self.G[aid]['w'].append(self.G[aid]['w'][0])    
            line=pufile.readline().strip() 
        pufile.close()
    
    
    def walker(self,start, walklen):
        walk=""
        ii=0        
        #start=random.randint(self.qnum+self.anum,self.N) # start from expert
        prev=start
        while ii<walklen: 
            #print("st="+ str(start)+" pre="+str(prev))            
            ind=0
            if len(self.G[start]['n'])==1:
                neib=self.G[start]['n']
                #print(neib)
                ind=0  
                if prev in neib:
                    indpre=neib.index(prev)                
                    del weights[indpre:indpre+1]
                    del neib[indpre:indpre+1]
                    #print(neib)
                    #print(weights)
                        
            if start<self.qnum or start>self.qnum+self.anum:
                if start>self.qnum+self.anum:
                    start=start-(self.anum)
                walk+= " "+str(start )           
            prev=start
            start=neib[ind]
            
            #if start>self.qnum+self.anum:
            ii+=1
        return walk.strip()    
    
    
    def get_train_pair(self,walks,windowsize, N): 
        #print(N)
        z=np.zeros((N))
        total=0
        for i in range(len(walks)):
            total+=len(walks[i])
            for j in walks[i]:
                z[int(j)]+=1
        #print(z) 
        #print(total)
        z1=z/total
        p=(np.sqrt(z1/0.001)+1)*(0.001/z1)  #probability of keeping a node in the traing
        #print(p)
        z2=np.power(z,.75)
        p2=z2/np.sum(z2)
        #print(p2)
        negsamples=[]
        for i in range(N):
            rep=int(p2[i]*100) 
            if rep==0:
               rep=1 
            for j in range(rep):
                negsamples.append(i) 
        #print(negsamples) 
        negs=np.array(negsamples)
        np.random.shuffle(negs)
        #print(negs)
        pairs=[]
        for i in range(len(walks)):
            walk=walks[i]                     
            for context_ind in range(len(walk)):            
                if context_ind>windowsize:
                    start=context_ind-windowsize
                else:
                    start=0
                for i in range(windowsize):
                    if i+start<context_ind:
                        x=np.random.randint(0,100)                    #print(x, p[int(walk[context_ind])]*100)
                        if (100-p[int(walk[context_ind])]*100)>x:
                            continue
                        if  walk[context_ind]!=walk[i+start] :  
                            pairs.append([int(walk[context_ind]),int(walk[i+start])])
                        if i+context_ind+1<len(walk):
                            x=np.random.randint(0,100)                    #print(x, p[int(walk[context_ind])]*100)
                            if (100-p[int(walk[context_ind])]*100)>x:
                                continue
                            if  walk[context_ind]!=walk[i+context_ind+1]:   
                                pairs.append([int(walk[context_ind]),int(walk[i+context_ind+1])])
        pairs=np.array(pairs)
        print("number of train samples:",len(pairs))
        return pairs,negs
    
    def get_train_pair2(self,walks,windowsize, N): 
        #print(N)
        z=np.zeros((N))
        total=0
        for i in range(len(walks)):
            total+=len(walks[i])
            for j in walks[i]:
                z[int(j)]+=1
        #print(z) 
        #print(total)
        z1=z/total
        p=(np.sqrt(z1/0.001)+1)*(0.001/z1)  #probability of keeping a node in the traing
        #print(p)
        z2=np.power(z,.75)
        p2=z2/np.sum(z2)
        #print(p2)
        negsamples=[]
        for i in range(N):
            rep=int(p2[i]*100)  
            for j in range(rep):
                negsamples.append(i) 
        #print(negsamples) 
        negs=np.array(negsamples)
        np.random.shuffle(negs)
        #print(negs)
        pairs=[]
        pairs=np.array(pairs)
        print("number of train samples:",len(pairs))
        return pairs,negs
    
    def walks(self,walklen,n1):
        G=nx.Graph();
        G=nx.read_weighted_edgelist(self.dataset+"/CQAG.txt")
        
        rw = BiasedRandomWalk(StellarGraph(G))

        weighted_walks = rw.run(
        nodes=G.nodes(), # root nodes
        length=walklen,    # maximum length of a random walk
        n=n1,          # number of random walks per root node 
        p=0.5,         # Defines (unormalised) probability, 1/p, of returning to source node
        q=2.0,         # Defines (unormalised) probability, 1/q, for moving away from source node
        weighted=True, #for weighted random walks
        seed=42        # random seed fixed for reproducibility
        )
        print("Number of random walks: {}".format(len(weighted_walks)))
        #print(weighted_walks[0:10])
        
        #remove answer nodes
        walks=[]
        for i in range(len(weighted_walks)):
            walk=weighted_walks[i]
            w=[]
            for node in walk:
                if int(node)<self.qnum:
                    w.append(node)
                elif int(node)>(self.qnum+self.anum):
                    n=int(node)-self.anum
                    w.append(str(n))
            walks.append(w)        
        print(walks[0:10])
        return walks
    
    def loss(predicted_y, target_y):        
        return tf.reduce_mean(tf.square(predicted_y - target_y))

    def model(self,inputs_i,inputs_j):    
        # look up embeddings for each term. [nbatch, qlen, emb_dim]
        i_embed = tf.nn.embedding_lookup(self.W1, inputs_i, name='iemb')
        j_embed = tf.nn.embedding_lookup(self.W2, inputs_j, name='jemb')          
        # Learning-To-Rank layer. o is the final matching score.
        temp=tf.transpose(j_embed, perm=[1, 0])
        o = tf.sigmoid(tf.matmul(i_embed, temp))
        o = tf.reshape(o, (len(o[0]),1))
        #print("o=")
        #print(o)
        return o
    
    def train(self, inputs_i, inputs_j, outputs, learning_rate):
        #print(inputs_i)
        i_embed = tf.nn.embedding_lookup(self.W1, inputs_i, name='iemb')
        j_embed = tf.nn.embedding_lookup(self.W2, inputs_j, name='jemb')  
        with tf.GradientTape() as t:
            current_loss = ExpertsEmbeding.loss(self.model(inputs_i,inputs_j), outputs)
        dW1, dW2 = t.gradient(current_loss, [self.W1, self.W2])        
        i_embed=i_embed-(learning_rate * dW1.values)
        
                
        k1=0
        #print(inputs_i.numpy())
        for k in inputs_i.numpy():
            #print(k)
            if k<self.qnum:
                self.W1[k,:].assign(i_embed[k1,:])
            else:
                teamcenter=np.array(self.teamcenters[self.ebt[self.anum+k][0]])
                c=tf.square(tf.subtract(i_embed,teamcenter))         
                d=tf.sqrt(tf.reduce_sum(c,axis=1)).numpy()[0]
                #print("d=",d)
                if d<self.teamoffsets[self.ebt[self.anum+k][0]][0]:
                    #print("offset=",self.teamoffsets[self.ebt[self.anum+k][0]][0])
                    self.W1[k,:].assign(i_embed[k1,:])
            k1+=1
        
        j_embed=j_embed-(learning_rate * dW2.values)
        #self.W2.assign(tf.tensor_scatter_nd_update(self.W2,indexw2,j_embed))
        k1=0
        #print(inputs_i.numpy())
        for k in inputs_j.numpy():
            #print(k)
            self.W2[k,:].assign(j_embed[k1,:])
            k1+=1
        #print(self.W1)
        #print(self.W2)
        return current_loss
    
        
    def run(self,walklen):
        #self.load_graph(dataset)        
        walks=self.walks(walklen,10) #n: number of walks start from a node
        #print(walks)
        pairs,negsamples=self.get_train_pair(walks,2,self.qnum+self.enum)
        lennegs=len(negsamples)
        print(pairs)
        epochs = range(2)
        loss_total=0
        train_len=len(pairs)
        logfile.close()
    def save_embeddings(self):
        #qfile=open(self.dataset+"/krnmdata1/teamsembeding.txt","w")
        w1=self.W1.numpy()
        w2=self.W2.numpy()
        np.savetxt(self.dataset+"/e2t/expert_question_w1_embedding.txt",w1, fmt='%f')
        np.savetxt(self.dataset+"/e2t/expert_question_w2_embedding.txt",w2, fmt='%f')

ob2=ExpertsEmbeding(32,data)
ob2.run(9)
