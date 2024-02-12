import numpy as np
from scipy.signal import cheby2,butter,sosfilt,firwin
from scipy import signal
import pyriemann
from pyriemann.estimation import Covariances
from pyriemann.utils.distance import *
from sklearn.metrics import silhouette_samples,silhouette_score
from pyriemann.classification import MDM,FgMDM
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from pyriemann.classification import MDM,FgMDM
from sklearn import metrics
from imblearn.metrics import specificity_score, sensitivity_score

def all_metrics(y_test,y_pred):
    acc=metrics.accuracy_score(y_test, y_pred)
    recall=metrics.recall_score(y_test,y_pred,average='micro') #average : string, [None, ‘binary’ (default), ‘micro’, ‘macro’, ‘samples’, ‘weighted’]
    precision=metrics.precision_score(y_test,y_pred,average='micro')
    f1=metrics.f1_score(y_test, y_pred,average='micro')
    kappa=metrics.cohen_kappa_score(y_test, y_pred)
    # tn, fp, fn, tp = metrics.confusion_matrix(y_test, y_pred).ravel()
    # sensitivity = tp / (tp+fn)
    # specificity = tn / (tn+fp)
    sensitivity=sensitivity_score(y_test, y_pred, average='micro')
    specificity=specificity_score(y_test, y_pred, average='micro')
    return acc,recall,precision,f1,kappa,sensitivity,specificity 

def dscore_dist_mean(dist_covs,mean_covs,y):
    n_class=mean_covs.shape[0]

    num1 = pairwise_distance(mean_covs)
    num2=num1[np.triu_indices(num1.shape[0], k = 1)] #upper trianglar above diag elements
    num=np.mean(num2) # inter class means 
    
    sd=[]
    for i in range(dist_covs.shape[1]):
        ind = np.where(y == i)[0]
        dist_i=dist_covs[ind,i]
        sd.append(np.mean(dist_i))
    sd=np.array(sd)
    den=np.mean(sd)  
         
    dscore_mdm=num/den
    return dscore_mdm

def get_covar(X):
    covar = Covariances(estimator='lwf').transform(X)
    return covar

def dscore(mats,ys):
    covs = get_covar(mats)
    n_clas=np.unique(ys).shape[0]
    mean_covs=[]
    for i in range(n_clas):
        ind = np.where(ys == i)[0]
        #print(ind)
        mean_covs.append(pyriemann.utils.mean.mean_riemann(covs[ind]))

    mean_covs = np.asarray(mean_covs)
    #print(mean_covs.shape)

    sd_covs=[]
    for i in range(n_clas):
        s=0
        ind = np.where(ys == i)[0]
        class_covs = covs[ind]
        n=class_covs.shape[0]
        for j in range(class_covs.shape[0]):
            s = s + distance_riemann(mean_covs[i],class_covs[j])
        s=s/n
        sd_covs.append(s)

    #print(sd_covs)

    num = pairwise_distance(mean_covs)
    num=np.sum(num)/2
    den = np.sum(sd_covs)/n_clas
    #print(num,den)
    d_score = num/den
    #print(d_score)
    return d_score

def gettw(startfreq=0,max_freq=4,bandwidth=[2,2.5,3,3.5,4]):
    # allfilters=[[8, 12], [12, 16], [16, 20], [20, 24], [24, 28], [28, 35]]
#     Keyword arguments:
#     bandwith -- numpy array containing bandwiths ex. [2,4,8,16,32]
#     Return:   numpy array of frequency bands
    
    f_bands = np.zeros((99,2)).astype(float)

    band_counter = 0
    for bw in bandwidth:
        startfreq = 0
        while (startfreq + bw <= max_freq): 
            f_bands[band_counter] = [startfreq, startfreq + bw]
            startfreq = startfreq + 0.5
#             if bw ==1.: # do 1Hz steps
#                 startfreq = startfreq +1
#             elif bw == 2.: # do 2Hz steps
#                 startfreq = startfreq +2 
#             else : # do 4 Hz steps if Bandwidths >= 4Hz
#                 startfreq = startfreq +int(bw/2)

            band_counter += 1 
    # convert array to normalized frequency 
    #     f_bands_nom = 2*f_bands[:band_counter]/f_s
    f_bands=np.array(f_bands)
    f_bands=f_bands[:band_counter,:]
#     print(f_bands)
    return f_bands

def get_timewin(tmin,tmax):
    bws=np.arange(2,tmax-tmin+0.5,0.5)
    tws=gettw(tmin,tmax,bws)
    return tws
    

def getfilters(startfreq=8,max_freq=30,bandwidth=[2,4]):
    # allfilters=[[8, 12], [12, 16], [16, 20], [20, 24], [24, 28], [28, 35]]
#     Keyword arguments:
#     bandwith -- numpy array containing bandwiths ex. [2,4,8,16,32]
#     Return:	numpy array of frequency bands
    
    f_bands = np.zeros((99,2)).astype(float)

    band_counter = 0
    for bw in bandwidth:
        startfreq = 4
        while (startfreq + bw <= max_freq): 
            f_bands[band_counter] = [startfreq, startfreq + bw]

            if bw ==1: # do 1Hz steps
                startfreq = startfreq +1
            elif bw == 2: # do 2Hz steps
                startfreq = startfreq +2 
            else : # do 4 Hz steps if Bandwidths >= 4Hz
                startfreq = startfreq +int(bw/2)

            band_counter += 1 
    # convert array to normalized frequency 
    #     f_bands_nom = 2*f_bands[:band_counter]/f_s
    f_bands=np.array(f_bands)
    f_bands=f_bands[:band_counter,:]
#     print(f_bands)
    return f_bands

def bestfilters(bands,x, y,nband_Sel): #dscore based selection
    mi_score=[]
    band=[]
    dband=[]
    for i in range(x.shape[-1]):
        b=bands[i]
        x1 = x[:,:,:,i]#butter_bandpass_filter(x, lowcut=b[0],highcut=b[1],fs=250,order=5)
        d=dscore(x1,y)
        mi_score.append(d)
        # print("band,dscore:",b,d)
        band.append(b)
        dband.append(d)
    n=nband_Sel
    mi_score=np.array(mi_score)
    indices = (-mi_score).argsort()[:n]
    bands=np.array(bands)
    ba=bands[indices]
    print("best bands:",ba)
    return ba,band,dband

def best_time(times,x,y,sfreq=250):
    # x = butter_bandpass_filter(x, lowcut=8,highcut=30,fs=250,order=5)
    mi_score=[]
    tim=[]
    dtim=[]
    for t in times:
        x1=x[:,:,int(t[0]*sfreq):int(t[1]*sfreq)]
        d=dscore(x1,y)
        mi_score.append(d)
        print("TW,dscore:",t,d)
        tim.append(t)
        dtim.append(d)
    maxpos = mi_score.index(max(mi_score)) 
    t=times[maxpos]
    print("best Time window:",t)
    return t,tim,dtim

def Sscore(x,y):
    covs = Covariances(estimator='lwf').transform(x)
    clf = MDM(metric=dict(mean='riemann',distance='riemann'))
    y_pred = clf.fit_predict(covs,y)
    dist_covs=clf.transform(covs)
    Sscore=silhouette_score(dist_covs, y)
    return Sscore
    
def bestfilters_Sscore(bands,x, y,nband_Sel): #dscore based selection
    mi_score=[]
    band=[]
    dband=[]
    for i in range(x.shape[-1]):
        b=bands[i]
        x1=x[:,:,:,i]
#         x1 = butter_bandpass_filter(x, lowcut=b[0],highcut=b[1],fs=sfreq,order=5)
        # print("filtered")
        d=Sscore(x1,y)
        mi_score.append(d)
        # print("band,dscore:",b,d)
        band.append(b)
        dband.append(d)
    n=nband_Sel
    mi_score=np.array(mi_score)
    indices = (-mi_score).argsort()[:n]
    bands=np.array(bands)
    ba=bands[indices]
    print("best bands:",ba)
    return ba,band,dband

def best_time_Sscore(times,x,y,sfreq):
#     x = butter_bandpass_filter(x, lowcut=4,highcut=40,fs=sfreq,order=5)
    # print("filtered")
    mi_score=[]
    tim=[]
    dtim=[]
    for t in times:
        x1=x[:,:,int(t[0]*sfreq):int(t[1]*sfreq)]
        d=Sscore(x1,y)
        mi_score.append(d)
        print("TW,Sscore:",t,d)
        tim.append(t)
        dtim.append(d)
    maxpos = mi_score.index(max(mi_score)) 
    t=times[maxpos]
    print("best Time window:",t)
    return t,tim,dtim

def label_map_fn(y):
    le = preprocessing.LabelEncoder()
    le.fit(y)
    le_label_map = dict(zip(le.classes_, le.transform(le.classes_)))
    return le, le_label_map

# def cal_Nfb_Sscore(dist_covs,y_train):
#     xcov=np.array(dist_covs)
#     x_sum_dist=[]
#     for i in range(xcov.shape[0]):
#         cov=xcov[i,:,:]
#         cov_sum=np.sum(cov, axis = 1)
#         x_sum_dist.append(cov_sum)
#     x_sum_dist=np.array(x_sum_dist).T
#     print(x_sum_dist.shape)
#     Nfb_Sscore=silhouette_score(x_sum_dist, y_train)
#     return Nfb_Sscore

# def Sscore_dist(x,y):
#     covs = Covariances(estimator='lwf').transform(x)
#     clf = MDM(metric=dict(mean='riemann',distance='riemann'))
#     y_pred = clf.fit_predict(covs,y)
#     dist_covs=clf.transform(covs)
#     Sscore=silhouette_score(dist_covs, y)
#     return Sscore,dist_covs

def cal_Nfb_Sscore(xcov,y_train):
    print("xcov",xcov.shape)
    x_sum_dist=[]
    for i in range(xcov.shape[1]):
        cov=xcov[:,i,:]
        cov_sum=np.mean(cov, axis = 0)
        x_sum_dist.append(cov_sum)
    x_sum_dist=np.array(x_sum_dist)
    print("x_sum_dist:",x_sum_dist.shape)
    Nfb_Sscore=silhouette_score(x_sum_dist, y_train)
    return Nfb_Sscore

def Sscore_dist(x,y):
    covs = Covariances(estimator='lwf').transform(x)
    clf = MDM(metric=dict(mean='riemann',distance='riemann'))
    y_pred = clf.fit_predict(covs,y)
    dist_covs=clf.transform(covs)
    Sscore=silhouette_score(dist_covs, y)
    return Sscore,dist_covs

def filters_score(bands,x,y):
    mi_score=[]
    band=[]
    dband=[]
    dist_covs=[]
    for i in range(x.shape[-1]):
        b=bands[i]
        x1=x[:,:,:,i]
        #x1 = butter_bandpass_filter(x, lowcut=b[0],highcut=b[1],fs=sfreq,order=5)
        # print("filtered")
        d,dist_cov=Sscore_dist(x1,y)
        print("band,Sscore",b,d)
        mi_score.append(d)
        dist_covs.append(dist_cov)
        band.append(b)
        dband.append(d)
    return mi_score, dist_covs, band, dband

def bestfilters_Nfb_Sscore(mi_score, dist_covs, band, dband,x, y,n): #dscore based selection
    mi_score=np.array(mi_score)
    indices = (-mi_score).argsort()[:n]
    band=np.array(band)
    ba=band[indices]
    dist_covs=np.array(dist_covs)
    dist_covs=dist_covs[indices,:,:]
    # print("dist_covs:",dist_covs.shape)
    Nfb_Sscore=cal_Nfb_Sscore(dist_covs,y)
    print("Nfb_Sscore:",Nfb_Sscore)
    print("best_band:",ba)
    return ba,band,dband,Nfb_Sscore

# def bestfilters_Nfb_Sscore0(bands,x, y,nband_Sel): #dscore based selection
#     mi_score=[]
#     band=[]
#     dband=[]
#     dist_covs=[]
#     for i in range(x.shape[-1]):
#         b=bands[i]
#         x1=x[:,:,:,i]
# #         x1 = butter_bandpass_filter(x, lowcut=b[0],highcut=b[1],fs=sfreq,order=5)
#         # print("filtered")
#         d,dist_cov=Sscore_dist(x1,y)
#         mi_score.append(d)
#         dist_covs.append(dist_cov)
#         # print("band,dscore:",b,d)
#         band.append(b)
#         dband.append(d)
#     n=nband_Sel
#     mi_score=np.array(mi_score)
#     indices = (-mi_score).argsort()[:n]
#     bands=np.array(bands)
#     ba=bands[indices]
#     Nfb_Sscore=cal_Nfb_Sscore(dist_covs,y)
#     print("best bands:",ba)
#     return ba,band,dband,Nfb_Sscore

def FB_to_cov(X,fb):
    xcov=fb.transform(X)
    x_sum_dist=[]
    for i in range(xcov.shape[0]):
        cov=xcov[i,:,:]
        cov_sum=np.sum(cov, axis = 1)
        x_sum_dist.append(cov_sum) 
    x_sum_dist=np.array(x_sum_dist)
    return x_sum_dist

def FB_MDM(X_train,y_train,X_test,y_test,fb):
    
    fb.fit(X_train,y_train)
    X_train_dist=FB_to_cov(X_train,fb)
    X_test_dist=FB_to_cov(X_test,fb)
    # print(X_test_dist.shape)
    y_pred=np.argmin(X_test_dist, axis=1)
    acc=accuracy_score(y_test,y_pred)
    return acc,y_pred

def FB_FgMDM(X_train,y_train,X_test,y_test,fb):
    
    fb.fit(X_train,y_train)
    X_train_dist=FB_to_cov(X_train,fb)
    X_test_dist=FB_to_cov(X_test,fb)
    y_pred=np.argmin(X_test_dist, axis=1)
    acc=accuracy_score(y_test,y_pred)
    return acc,y_pred

def dscore_mdm(X_train,y_train):  
    covs = Covariances(estimator='lwf').transform(X_train)
    clf = MDM(metric=dict(mean='riemann',distance='riemann'))
    clf.fit(covs,y_train)
    mean_covs=np.array(clf.covmeans_)
    n_class=mean_covs.shape[0]
    print("mdm meancovs:",mean_covs.shape)
    num1 = pairwise_distance(mean_covs)
    num2=num1[np.triu_indices(num1.shape[0], k = 1)] #upper trianglar above diag elements
    num=np.mean(num2) # inter class means
    
    dist=clf.transform(covs)
    print("mdm distcovs:",dist.shape)
    sd=[]
    for i in range(dist.shape[1]):
        ind = np.where(y_train == i)[0]
        dist_i=dist[ind,i]
        sd.append(np.mean(dist_i))
    sd=np.array(sd)
    den=np.mean(sd) 
         
    dscore_mdm=num/den
    print("dscore_mdm:",num,den,dscore_mdm)
    return dscore_mdm

def dscore_dist(X_train,y_train):
    covs = Covariances(estimator='lwf').transform(X_train)
    clf = MDM(metric=dict(mean='riemann',distance='riemann'))
    clf.fit(covs,y_train)
    mean_covs=np.array(clf.covmeans_)
    n_class=mean_covs.shape[0]
    # print("mdm meancovs:",mean_covs.shape)
    
    num1 = pairwise_distance(mean_covs)
    num2=num1[np.triu_indices(num1.shape[0], k = 1)] #upper trianglar above diag elements
    num=np.mean(num2) # inter class means 
    
    dist=clf.transform(covs)
    
    sd=[]
    for i in range(dist.shape[1]):
        ind = np.where(y_train == i)[0]
        dist_i=dist[ind,i]
        sd.append(np.mean(dist_i))
    sd=np.array(sd)
    den=np.mean(sd)  
         
    dscore_mdm=num/den
    print("dscore_dist:",num,den,dscore_mdm)
    return dscore_mdm, mean_covs, dist

def filters_dscore(bands,x,y):
    mi_score=[]
    band=[]
    dband=[]
    dist_covs=[]
    mean_covs=[]
    for i in range(x.shape[-1]):
        b=bands[i]
        x1=x[:,:,:,i]
        #x1 = butter_bandpass_filter(x, lowcut=b[0],highcut=b[1],fs=sfreq,order=5)
        # print("filtered")
        d,mean_cov,dist_cov=dscore_dist(x1,y)
        print("band,dscore",b,d)
        mi_score.append(d)
        dist_covs.append(dist_cov)
        mean_covs.append(mean_cov)
        band.append(b)
        dband.append(d)
    return mi_score, mean_covs,dist_covs, band, dband

def mean_covs_1(mean_covs):
    # print("mean_covs:",mean_covs.shape) # n_fb,n_class,ch,ch
    mean_cov=[]
    for i in range(mean_covs.shape[1]):
        cov=mean_covs[:,i,:,:] # n_fb,ch,ch
        print("cov:",cov.shape)
        cov_mean=pyriemann.utils.mean.mean_riemann(cov)
        print("cov_mean:",cov_mean.shape)
        mean_cov.append(cov_mean)
    mean_cov=np.array(mean_cov)
    # print("mean_cov:",mean_cov.shape)
    return mean_cov #n_class,ch,ch

def dist_covs_1(dist_covs):
    # print("dist_covs:",dist_covs.shape) # n_fb,trials,n_class
    dist_cov=[]
    for i in range(dist_covs.shape[1]):
        cov=dist_covs[:,i,:]
        cov_sum=np.mean(cov, axis = 0)
        dist_cov.append(cov_sum)
    dist_cov=np.array(dist_cov)
    # print("dist_cov:",dist_cov.shape)
    return dist_cov #trials,n_class

def mean_covs_2(mean_covs):
    print("mean_covs_2")
    # print("mean_covs:",mean_covs.shape) # n_fb,n_class,ch,ch
    num_lst=[]
    for i in range(mean_covs.shape[0]):
        cov=mean_covs[i,:,:,:] # n_class,ch,ch
        d_mat = pairwise_distance(cov)
        d_arr=d_mat[np.triu_indices(d_mat.shape[0], k = 1)] #upper trianglar above diag elements
        num_lst.append(np.mean(d_arr)) # inter class means 
    num_lst=np.array(num_lst)
    num=np.mean(num_lst)
    return num #n_class,ch,ch

def mean_covs_3(mean_covs):
    print("mean_covs_3") # n_fb,n_class,ch,ch
    return 1 #n_class,ch,ch

def cal_Nfb_dscore(mean_covs,dist_covs,y_train):
    
    # mean_cov=mean_covs_1(mean_covs)
    mean_cov=mean_covs_1(mean_covs)
    dist_cov=dist_covs_1(dist_covs)
    ### num ###
    num1 = pairwise_distance(mean_cov)
    num2=num1[np.triu_indices(num1.shape[0], k = 1)] #upper trianglar above diag elements
    num=np.mean(num2) # inter class means 
    ### den ###   
    sd=[]
    for i in range(dist_cov.shape[1]):
        ind = np.where(y_train == i)[0]
        dist_i=dist_cov[ind,i]
        sd.append(np.mean(dist_i))
    sd=np.array(sd)
    den=np.mean(sd)
    ### dscore ###     
    Nfb_dscore=num/den
    
    return Nfb_dscore

def bestfilters_Nfb_dscore(mi_score, mean_covs,dist_covs, band, dband,x, y,n): #dscore based selection
    mi_score=np.array(mi_score)
    indices = (-mi_score).argsort()[:n]
    band=np.array(band)
    ba=band[indices]
    
    mean_covs=np.array(mean_covs)
    mean_covs=mean_covs[indices,:,:]
    # print("mean_covs:",mean_covs.shape)
    
    dist_covs=np.array(dist_covs)
    dist_covs=dist_covs[indices,:,:]
    # print("dist_covs:",dist_covs.shape)
    
    Nfb_dscore=cal_Nfb_dscore(mean_covs,dist_covs,y)
    print("Nfb_dscore:",Nfb_dscore)
    print("best_band:",ba)
    return ba,band,dband,Nfb_dscore

def bestfilters_dscoredm(bands,x,y,nband_Sel):
    mi_score=[]
    band=[]
    dband=[]
    for i in range(x.shape[-1]):
        b=bands[i]
        x1 = x[:,:,:,i]#butter_bandpass_filter(x, lowcut=b[0],highcut=b[1],fs=250,order=5)
        covs = Covariances(estimator='lwf').transform(x1)
        clf = MDM(metric=dict(mean='riemann',distance='riemann'))
        y_pred = clf.fit_predict(covs,y)
        dist_covs=clf.transform(covs)
        mean_covs=np.array(clf.covmeans_)
        d=dscore_dist_mean(dist_covs,mean_covs,y)
        mi_score.append(d)
        # print("band,dscore:",b,d)
        band.append(b)
        dband.append(d)
    n=nband_Sel
    mi_score=np.array(mi_score)
    indices = (-mi_score).argsort()[:n]
    bands=np.array(bands)
    ba=bands[indices]
    # print("best bands:",ba)
    return ba,band,dband

def best_time_dscoredm(times,x,y,sfreq):
#     x = butter_bandpass_filter(x, lowcut=4,highcut=40,fs=sfreq,order=5)
    # print("filtered")
    mi_score=[]
    tim=[]
    dtim=[]
    for t in times:
        x1=x[:,:,int(t[0]*sfreq):int(t[1]*sfreq)]
        covs = Covariances(estimator='lwf').transform(x1)
        clf = MDM(metric=dict(mean='riemann',distance='riemann'))
        y_pred = clf.fit_predict(covs,y)
        dist_covs=clf.transform(covs)
        mean_covs=np.array(clf.covmeans_)
        d=dscore_dist_mean(dist_covs,mean_covs,y)
        mi_score.append(d)
        print("TW,Sscore:",t,d)
        tim.append(t)
        dtim.append(d)
    maxpos = mi_score.index(max(mi_score)) 
    t=times[maxpos]
    print("best Time window:",t)
    return t,tim,dtim

def ALLscore_dist(x,y):
    score={}
    covs = Covariances(estimator='lwf').transform(x)
#     p_covs=pairwise_distance(covs)
    
#     score["silhouette_score"]=silhouette_score(p_covs, y, metric = 'precomputed')
#     score["silhouette_score p_covs"]=silhouette_score(p_covs, y)
#     score["calinski_harabasz_score p_covs"]=calinski_harabasz_score(p_covs, y)
#     score["davies_bouldin_score p_covs"]=davies_bouldin_score(p_covs, y) #small is better
    
#     score["dscore"]=dscore(covs,y)
    
    clf = MDM(metric=dict(mean='riemann',distance='riemann'))
    y_pred = clf.fit_predict(covs,y)
    dist_covs=clf.transform(covs)
    mean_covs=np.array(clf.covmeans_)
#     score["silhouette_score dist_covs"]=silhouette_score(dist_covs, y)
#     score["calinski_harabasz_score dist_covs"]=calinski_harabasz_score(dist_covs, y)
#     score["davies_bouldin_score dist_covs"]=davies_bouldin_score(dist_covs, y) #small is better

    score["dscore_dist_mean"]=dscore_dist_mean(dist_covs,mean_covs,y)
    
    return score,covs,dist_covs,mean_covs

def filters_ALLscore(bands,x,y):
    all_scores=[]
    band=[]
    dband=[]
    dist_covs=[]
    mean_covs=[]
    covs=[]
    
    for i in range(x.shape[-1]):
        b=bands[i]
        x1=x[:,:,:,i]
        #x1 = butter_bandpass_filter(x, lowcut=b[0],highcut=b[1],fs=sfreq,order=5)
        # print("filtered")
        score,cov,dist_cov,mean_cov=ALLscore_dist(x1,y)
#         print("band,Sscore",b,d)
        all_scores.append(score)
        dist_covs.append(dist_cov)
        mean_covs.append(mean_cov)
        covs.append(cov)
        band.append(b)
        dband.append(score)
    return all_scores, dist_covs,mean_covs,covs, band

def extract_listdict(all_scores,key_score):
    mi_score=[]
    for i in range(len(all_scores)):
        sdict=all_scores[i]
        mi_score.append(sdict[key_score])
    return mi_score

def cal_Nfb_dscoredm(mean_covs,covs,dist_covs,y,key_score):
    dist_covs=np.array(dist_covs);     dist_covs=np.resize(dist_covs,(dist_covs.shape[0]*dist_covs.shape[1],dist_covs.shape[2]))
    mean_covs=np.array(mean_covs);    mean_covs=np.resize(mean_covs,(mean_covs.shape[0]*mean_covs.shape[1],mean_covs.shape[2],mean_covs.shape[3]))
    covs=np.array(covs);    covs=np.resize(covs,(covs.shape[0]*covs.shape[1],covs.shape[2],covs.shape[3]))
    y=np.array(y);  y=np.resize(y,(covs.shape[0]))
#     p_covs=pairwise_distance(covs)
    #print("p_covs:",p_covs.shape,"|| y:",np.unique(y))
#     score=0
    if key_score == "silhouette_score":
        score=silhouette_score(p_covs, y, metric = 'precomputed')
    elif key_score == "silhouette_score p_covs":
        score=silhouette_score(p_covs, y)
    elif key_score == "calinski_harabasz_score p_covs":
        score=calinski_harabasz_score(p_covs, y)
    elif key_score == "dscore":
        score=dscore(covs,y)
    elif key_score == "silhouette_score dist_covs":
        score=silhouette_score(dist_covs, y)
    elif key_score == "calinski_harabasz_score dist_covs":
        score=calinski_harabasz_score(dist_covs, y)
    elif key_score == "dscore_dist_mean":
        score=dscore_dist_mean(dist_covs,mean_covs,y)  
    return score

def bestfilters_Nfb_dscoredm(all_scores,dist_covs,mean_covs,covs,bands, y,n,key_scores): #dscore based selection
    Nfb_ALLscore={};best_bands={}
    bands=np.array(bands);  dist_covs=np.array(dist_covs); mean_covs=np.array(mean_covs);covs=np.array(covs); 
    for key_score in key_scores:
        mi_score=extract_listdict(all_scores,key_score)
        mi_score=np.array(mi_score);        indices = (-mi_score).argsort()[:n]
        best_bands[key_score]=bands[indices]
        dist_cov=dist_covs[indices,:,:]
        mean_cov=mean_covs[indices,:,:]
        cov=covs[indices,:,:]
        Nfb_ALLscore[key_score]=cal_Nfb_dscoredm(mean_cov,cov,dist_cov,y,key_score)
#         print(key_score,"Nfb_score:",Nfb_ALLscore[key_score])
#         print(key_score,"best_bands:",best_bands[key_score])
    
    return best_bands,Nfb_ALLscore 

def get_bestclasses(X0,y0,class_comb_list):
#     print(X0.shape,y0.shape,class_comb_list)
    dscore_allclass=[]
    for class_comb in class_comb_list:
        value1=class_comb[0];value2=class_comb[1]
#         print(value1,value2)
        ind0 = [i for i, value in enumerate(y0) if ((value == value1) or (value ==value2))]
        X_train=X0[ind0];y_train =y0[ind0]
        covs = Covariances(estimator='lwf').transform(X_train)
        clf = MDM(metric=dict(mean='riemann',distance='riemann'))
        y_pred = clf.fit_predict(covs,y_train)
        dist_covs=clf.transform(covs)
        mean_covs=np.array(clf.covmeans_)
        le1,le_label_map1=label_map_fn(y_train)
        y_train=le1.transform(y_train)
#         print(X_train.shape,y_train.shape,np.unique(y_train))
        d=dscore_dist_mean(dist_covs,mean_covs,y_train)
        dscore_allclass.append(d)
        print("class_comb,dscore:",class_comb,d)
    maxpos = dscore_allclass.index(max(dscore_allclass)) 
    bestclass=class_comb_list[maxpos]
    print("best class_comb:",bestclass)
    return bestclass,class_comb_list,dscore_allclass

# bestclass,allclass,dscore_allclass=get_bestclasses(X0,y0,class_comb_list)

def get_dscore_class(X_train,y_train):
    covs = Covariances(estimator='lwf').transform(X_train)
    clf = MDM(metric=dict(mean='riemann',distance='riemann'))
    y_pred = clf.fit_predict(covs,y_train)
    dist_covs=clf.transform(covs)
    mean_covs=np.array(clf.covmeans_)
    le1,le_label_map1=label_map_fn(y_train)
    y_train=le1.transform(y_train)
#     print(X_train.shape,y_train.shape,np.unique(y_train))
    d=dscore_dist_mean(dist_covs,mean_covs,y_train)
    
    return d