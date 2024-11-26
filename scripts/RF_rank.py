import pandas  as pd
import numpy as np
import sys,pickle,argparse,os,subprocess
import math

#from collections import  Counter
#from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
#from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import matthews_corrcoef,roc_curve,auc
from sklearn.metrics import confusion_matrix,accuracy_score,f1_score,roc_auc_score,recall_score,precision_score

#from sklearn.model_selection import LeaveOneOut

#import xgboost as xgb
#from sklearn.model_selection import KFold #回归问题
#from sklearn.model_selection import GridSearchCV #网格搜索

def ACC(y_true,y_pred):
  x=0
  for i  in range(y_pred.shape[0]):
    a=y_true[i,]
    b=y_pred[i,]
    if (b==a):
      x=x+1
  return format(x/y_pred.shape[0]*100,'0.4f')

def MCC(y_true,y_pred):
  x=matthews_corrcoef(y_true, y_pred)
  return format(x*100,'0.4f')

def cal_score(y_true,y_pred):
  yt=[];yp=[];
  for i in range(y_pred.shape[0]):
    if y_true[i]=="R":
      yt.append(1)
    else:
      yt.append(0)
    if y_pred[i]=="R":
      yp.append(1)
    else:
      yp.append(0)
  precision=format(precision_score(yt,yp)*100,'0.4f')
  sensitivity=format(recall_score(yt,yp)*100,'0.4f')    ##recall
  f1=format(f1_score(yt,yp)*100,'0.4f')
  if confusion_matrix(yt,yp).shape[0]==1 and yt[0]==1:
    specificity=0
  elif confusion_matrix(yt,yp).shape[0]==1 and yt[0]==0:
    specificity=100
  else:
    tn, fp, fn, tp = confusion_matrix(yt,yp).ravel()
    specificity=format(tn/(tn+fp)*100,'0.4f')
  return precision,sensitivity,specificity,f1

def me_vme(y_true,y_pred):
  l=len(y_pred)
  me_num=0;vme_num=0;
  r_num=0;s_num=0;
  for i in range(l):
    if y_true[i]=="R":
       r_num +=1
    if y_true[i]=="S":
       s_num +=1

    if y_true[i]=="R" and y_pred[i]=="S":               ##R预测为S
      vme_num +=1
    if y_true[i]=="S" and y_pred[i]=="R":               ##S预测为R
      me_num +=1
  me=format(me_num/s_num*100,'0.4f')
  vme=format(vme_num/r_num*100,'0.4f')
  return me,vme
def plusdict(A,B): #(plus B to A)
  for key in B:
    if key in A:
      A[key]=A[key]+B[key]
    else:
      A[key]=B[key]
  return A

def forestModel(X,Y):
  forest = RandomForestClassifier(n_estimators=n_estimators, max_features='sqrt',n_jobs=-1)
  forest.fit(X, Y)
  return forest

def forestFeature(X,Y,drug):
  forest=forestModel(X,Y)
  importance=forest.feature_importances_
  feature=pd.DataFrame(np.array(importance),columns=[drug],index=X.columns).sort_values(by=[drug],ascending=False)
  return feature

def forestPredict(X,Y,x):
  forest=forestModel(X,Y)
  y=forest.predict(x)
  return y

##################common sub function
def mkdir_or_die(dir_to_make):
    target_dir = os.path.abspath(dir_to_make)
    if not os.path.isdir(target_dir):
        try:
            os.makedirs(target_dir)
        except FileExistsError:  # in case of Race Condition
            pass
    return target_dir

def run_cmd(cmd):
    print("Begin running command : "+cmd+"\n")
    run_cmd = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    return run_cmd.returncode
 
def __main__():
  description = "This script is used to filter tumor SNP/indel existed in normal sample!\n Contact with: wenyanhua@hugobiotech.com\n"
  quick_usage= 'python ' + sys.argv[0] + ' -id idir -od odir'
  newParser = argparse.ArgumentParser( description = description, usage = quick_usage );
  newParser.add_argument( "-sir", dest="sir", help="", required=True);
  newParser.add_argument( "-matrix", dest="matrix", help="", required=True);
  newParser.add_argument( "-od", dest="od", help="", required=True);
  newParser.add_argument( "-n", dest="n", help="n_estimators, default sqrt(feature_num)*10");
  newParser.add_argument( "-drug", dest="drug", help="you can set multiple drugs separated by comma");
  newParser.add_argument( "-times", dest="times", help="train num such as 10, train stands for train and test 7:3 for 10 times",default=10);

  args = newParser.parse_args();
  if not args.sir or not args.matrix or not args.od:
    newParser.print_help()
    sys.exit(1)
  global n_estimators

  od=mkdir_or_die(args.od)
  sir=pd.read_csv(args.sir,sep="\t",index_col=0,header=0)
  sir.index=list(map(str,sir.index.tolist()))
  if args.drug:
    drugs=args.drug.split(',')
  else:
    drugs=sir.columns
 
  data=pd.read_csv(args.matrix,sep="\t",index_col=0,header=0)
  num=data.shape[0]
  if args.n:
    n_estimators=int(args.n)
  else:
    n_estimators=10*(math.ceil(math.sqrt(num)))

  for drug in drugs:
    print(drug)
    odir=mkdir_or_die(od+'/'+drug);
    rdir=mkdir_or_die(od+'/'+drug+'.rank')
    samples=sir[sir[drug]!="I"][drug].dropna().index.tolist()
    samples=list(set(samples).intersection(set(data.columns)))
    X=data[samples].T;
    Y=sir.loc[samples][drug];

    print(X)
    print(Y)
##删了一句SR小于5 continue的命令行
    acc=[];mcc=[];pre=[];recall=[];spe=[];f1s=[];mes=[];vmes=[];
    rank={};
    loo = StratifiedShuffleSplit(n_splits=int(args.times), test_size=0.3, random_state=64)
    print("loo done.")
    i=0
    odrug=od+'/'+drug+'.acc_mcc.xls'
    fd=open(odrug,'w')
    fd.write("Drug\tRank\tACC\tMCC\tRECALL\tSpecificity\tF1\tPrecision\tME\tVME\n")
    opre=odir+'/'+drug
    for train_index, test_index in loo.split(X,Y):
      i+=1
      print("loo.split")
      X_train, X_test = X.iloc[train_index], X.iloc[test_index]
      Y_train, Y_test = Y[train_index], Y[test_index]
      X_train.to_csv(opre+'.train.X.'+str(i),sep="\t",index_label="Sample")
      X_test.to_csv(opre+'.test.X.'+str(i),sep="\t",index_label="Sample")
      Y_train.to_csv(opre+'.train.Y.'+str(i),sep="\t",index_label="Sample")
      Y_test.to_csv(opre+'.test.Y.'+str(i),sep="\t",index_label="Sample")
      print(Y_train)
      feature = forestFeature(X_train,Y_train,drug)
      feature['rank']=feature.rank(method="dense",ascending=False)
      feature.to_csv(rdir+'/rank.'+str(i),sep="\t",index_label="ID")
      dic =dict(zip(feature.index,feature['rank']))
      rank=plusdict(rank,dic)
      genes=feature[:num].index.tolist()
      xtrain,xtest=X_train[genes],X_test[genes]
      test_pre=forestPredict(xtrain,Y_train,xtest)
      realpre =pd.DataFrame({'real':np.array(Y_test),'predict':np.array(test_pre)},index=Y_test.index)
      realpre.to_csv(opre+'.test.Y_predict.'+str(i),sep="\t",index_label="Sample")

      a=ACC(np.array(Y_test),np.array(test_pre))
      m=MCC(np.array(Y_test),np.array(test_pre))
      precision,sensitivity,specificity,f1=cal_score(np.array(Y_test),np.array(test_pre))
      me, vme = me_vme(np.array(Y_test),np.array(test_pre))
      mes.append(me);vmes.append(vme)
      acc.append(a);mcc.append(m);pre.append(precision)
      recall.append(sensitivity);spe.append(specificity);f1s.append(f1)
      fd.write("%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n"%(drug,i,a,m,sensitivity,specificity,f1,precision,me,vme))
    rk=pd.DataFrame([rank]).T.sort_values(by=[0],ascending=True)
    rk[0]=rk[0]/len(acc)
    rk.to_csv(od+'/'+drug+".genes_rank.txt",sep="\t",header=['Mean_rank'],index_label="Gene")
    fd.write("%s\tmean\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n"%(drug,format(np.mean(list(map(float,acc))),'0.4f'),format(np.mean(list(map(float,mcc))),'0.4f'),format(np.mean(list(map(float,recall))),'0.4f'),format(np.mean(list(map(float,spe))),'0.4f'),format(np.mean(list(map(float,f1s))),'0.4f'),format(np.mean(list(map(float,pre))),'0.4f'),format(np.mean(list(map(float,mes))),'0.4f'),format(np.mean(list(map(float,vmes))),'0.4f')))    
    fd.close()
  
'''
GridSearchCV(cv=10,
             estimator=XGBClassifier(base_score=None, booster=None,
                                     colsample_bylevel=None,
                                    colsample_bynode=None,
                                     colsample_bytree=None,
                                     enable_categorical=False, gamma=None,
                                     gpu_id=None, importance_type=None,
                                     interaction_constraints=None,
                                     learning_rate=None, max_delta_step=None,
                                     max_depth=None, min_child_weight=None,
                                     missing=nan, monotone_constraints=None,
                                     n_estimators=100, n_jobs=None,
                                     num_parallel_tree=None, predictor=None,
                                     random_state=None, reg_alpha=None,
                                     reg_lambda=None, scale_pos_weight=None,
                                     subsample=None, tree_method=None,
                                     validate_parameters=None, verbosity=None),
'''
if __name__ == "__main__": __main__()

