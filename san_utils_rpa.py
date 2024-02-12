# get the functions from RPA package
import transfer_learning as TL
import diffusion_map as DM
import get_dataset as GD
from pyriemann.utils.mean import mean_riemann, geodesic_riemann
from pyriemann.utils.base import invsqrtm, sqrtm, logm, expm, powm
from pyriemann.estimation import Covariances
import numpy as np

def parallel_transport_covariance_matrix(C, R):

    return np.dot(invsqrtm(R), np.dot(C, invsqrtm(R)))

def parallel_transport_covariances(C, R):

    Cprt = []
    for Ci, Ri in zip(C, R):
        Cprt.append(parallel_transport_covariance_matrix(Ci, Ri))
    return np.stack(Cprt)


def transform_org2rct(source_org, target_org_train):

    source_rct = {}
    T_source = np.stack([mean_riemann(source_org['covs'])] * len(source_org['covs']))
    source_rct['covs'] = parallel_transport_covariances(source_org['covs'], T_source)
    source_rct['labels'] = source_org['labels']

    M_target = mean_riemann(target_org_train['covs']) ### changed here

    target_rct_train = {}
    T_target_train = np.stack([M_target]*len(target_org_train['covs']))
    target_rct_train['covs'] = parallel_transport_covariances(target_org_train['covs'], T_target_train)
    target_rct_train['labels'] = target_org_train['labels']
    
    return source_rct, target_rct_train

def no_transform_baseline(X_train, X_test, y_train, y_test,target_train_split=10):
    # formt X & y
    source, target=format_xy_rpa(X_train, X_test, y_train, y_test)
    X_train, y_train = source['covs'], source['labels']
    X_test, y_test = target['covs'], target['labels'] 
          
    return X_train, X_test, y_train, y_test

def calib_transform(X_train, X_test, y_train, y_test,target_train_split=10):
    # formt X & y
    source, target=format_xy_rpa(X_train, X_test, y_train, y_test)
 
    # get the split for the source and target dataset
    source_org, target_org_train, target_org_test = TL.get_sourcetarget_split(source, target, ncovs_train=target_train_split)

    #print(source_train["covs"].shape, target_train["covs"].shape, source_train["labels"].shape, target_train["labels"].shape)
    # reformt to X & y
    X_train, X_test, y_train, y_test=reformat_rpa_xy(source_org, target_org_train, target_org_test)       
    return X_train, X_test, y_train, y_test

def fb_to_oneband(source_org, i):
    s={}
    s["covs"]=source_org["covs"][:,:,:,i]
    s["labels"]=source_org["labels"]
    return s
    

def RCT_transform(X_train, X_test, y_train, y_test,target_train_split=10): # 10% data from target session to train):
    # formt X & y
    source, target=format_xy_rpa(X_train, X_test, y_train, y_test)
    source_cov=[] 
    target_train_cov=[]
    target_test_cov=[]
    source_train={}
    target_train={}
    target_test={}
    # get the split for the source and target dataset
    source_org, target_org_train, target_org_test = TL.get_sourcetarget_split(source, target, ncovs_train=target_train_split)
        

    for i in range(X_train.shape[-1]):
        ## splitting covs & labels
        source_org1=fb_to_oneband(source_org, i)
        target_org_train1=fb_to_oneband(target_org_train, i)
        target_org_test1=fb_to_oneband(target_org_test, i)
        # get the score with the re-centered matrices
#         source_rct, target_rct_train, target_rct_test = TL.RPA_recenter(source_org1, target_org_train1, target_org_test1)
        source_rct, target_rct_train, target_rct_test = transform_org2rct(source_org1, target_org_train1, target_org_test1)
        transform_org2rct
        #append for all filters
        source_cov.append(source_rct["covs"])
        target_train_cov.append(target_rct_train["covs"])
        target_test_cov.append(target_rct_test["covs"])
    # grouping covs & labels
    source_train["covs"]=np.moveaxis(np.array(source_cov), 0, -1)
    target_train["covs"]=np.moveaxis(np.array(target_train_cov), 0, -1)
    target_test["covs"]=np.moveaxis(np.array(target_test_cov), 0, -1)
    source_train["labels"]=source_rct["labels"]
    target_train["labels"]=target_rct_train["labels"]
    target_test["labels"]=target_rct_test["labels"]
#     print(source_train["covs"].shape, target_train["covs"].shape, source_train["labels"].shape, target_train["labels"].shape)
    # reformt to X & y
    X_train, X_test, y_train, y_test=reformat_rpa_xy(source_train, target_train, target_test)       
    return X_train, X_test, y_train, y_test

def RCT_transform_nosplit(X_train, X_test, y_train, y_test): # 10% data from target session to train):
    # formt X & y
    source, target=format_xy_rpa(X_train, X_test, y_train, y_test)
    source_cov=[] 
    target_cov=[]
    source_train={}
    target_train={}
    for i in range(X_train.shape[-1]):
        ## splitting covs & labels
        source_org1=fb_to_oneband(source, i)
        target_org1=fb_to_oneband(target, i)
        # get the score with the re-centered matrices
        source_rct, target_rct = transform_org2rct(source_org1, target_org1)
        #append for all filters
        source_cov.append(source_rct["covs"])
        target_cov.append(target_rct["covs"])
    # grouping covs & labels
    source_train["covs"]=np.moveaxis(np.array(source_cov), 0, -1)
    target_train["covs"]=np.moveaxis(np.array(target_cov), 0, -1)
    source_train["labels"]=source_rct["labels"]
    target_train["labels"]=target_rct["labels"]
#     print(source_train["covs"].shape, target_train["covs"].shape, source_train["labels"].shape, target_train["labels"].shape)
    # reformt to X & y
    X_train, y_train = source_train['covs'], source_train['labels']
    X_test, y_test = target_train['covs'], target_train['labels']      
    return X_train, X_test, y_train, y_test

def RPA_transform(X_train, X_test, y_train, y_test,target_train_split=10): # 10% data from target session to train):
    # formt X & y
    source, target=format_xy_rpa(X_train, X_test, y_train, y_test)
    source_cov=[] 
    target_train_cov=[]
    target_test_cov=[]
    source_train={}
    target_train={}
    target_test={}
    # get the split for the source and target dataset
    source_org, target_org_train, target_org_test = TL.get_sourcetarget_split(source, target, ncovs_train=target_train_split)
        

    for i in range(X_train.shape[-1]):
        ## splitting covs & labels
        source_org1=fb_to_oneband(source_org, i)
        target_org_train1=fb_to_oneband(target_org_train, i)
        target_org_test1=fb_to_oneband(target_org_test, i)
        # get the score with the re-centered matrices
        source_rct, target_rct_train, target_rct_test = TL.RPA_recenter(source_org1, target_org_train1, target_org_test1)
        # rotate the re-centered-stretched matrices using information from classes
        source_rpa, target_rpa_train, target_rpa_test = TL.RPA_rotate(source_rct, target_rct_train, target_rct_test)
        #append for all filters
        source_cov.append(source_rpa["covs"])
        target_train_cov.append(target_rpa_train["covs"])
        target_test_cov.append(target_rpa_test["covs"])
    # grouping covs & labels
    source_train["covs"]=np.moveaxis(np.array(source_cov), 0, -1)
    target_train["covs"]=np.moveaxis(np.array(target_train_cov), 0, -1)
    target_test["covs"]=np.moveaxis(np.array(target_test_cov), 0, -1)
    source_train["labels"]=source_rpa["labels"]
    target_train["labels"]=target_rpa_train["labels"]
    target_test["labels"]=target_rpa_test["labels"]
#     print(source_train["covs"].shape, target_train["covs"].shape, source_train["labels"].shape, target_train["labels"].shape)
    # reformt to X & y
    X_train, X_test, y_train, y_test=reformat_rpa_xy(source_train, target_train, target_test)       
    return X_train, X_test, y_train, y_test

def covar_filterbank(X):
    covar=[]
    for i in range(X.shape[-1]):
        covar.append(Covariances(estimator='oas').fit_transform(X[:,:,:,i]))
    covar_X=np.array(covar)
    covar_X=np.moveaxis(covar_X, 0, -1)
#     print(covar_X.shape)
    return covar_X

def format_xy_rpa(X_train, X_test, y_train, y_test):
    source={}
    target={}
    source["covs"]=covar_filterbank(X_train)
    source["labels"]=y_train
    target["covs"]=covar_filterbank(X_test)
    target["labels"]=y_test
#     print(source["covs"].shape)
#     print(source["labels"].shape)
#     print(target["covs"].shape)
#     print(target["labels"].shape)
    return source, target

def reformat_rpa_xy(source, target_train, target_test):
    covs_source, y_source = source['covs'], source['labels']
    covs_target_train, y_target_train = target_train['covs'], target_train['labels']
    covs_target_test, y_target_test = target_test['covs'], target_test['labels']

    covs_train = np.concatenate([covs_source, covs_target_train])
    y_train = np.concatenate([y_source, y_target_train])
    return covs_train,covs_target_test,y_train, y_target_test