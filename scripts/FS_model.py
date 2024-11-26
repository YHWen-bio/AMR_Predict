import pandas  as pd
import numpy as np
import sys,pickle,argparse,os,subprocess
import math
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

from collections import  Counter
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import matthews_corrcoef,roc_curve,auc
from sklearn.metrics import confusion_matrix,accuracy_score,f1_score,roc_auc_score,recall_score,precision_score

from sklearn.model_selection import LeaveOneOut

import xgboost as xgb
from sklearn.model_selection import KFold #回归问题
from sklearn.model_selection import GridSearchCV #网格搜索

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
#  me=format(me_num/len(y_true)*100,'0.4f')
#  vme=format(vme_num/len(y_true)*100,'0.4f')
  return me,vme

#for font in fm.fontManager.ttflist:
#  print(font.name)
plt.rcParams['font.family']=['WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus']=False

def plotauc(opng,drug,fpr,tpr,auc):
  plt.figure()
  plt.title(drug)
  plt.plot(fpr,tpr,'b',label="AUC = %0.2f"% auc)
  plt.legend(loc="lower right")
  plt.ylabel("True Positive Rate")
  plt.xlabel("False Positive Rate")
  plt.gcf().savefig(opng)
  return

def plusdict(A,B): #(plus B to A)
  for key in B:
    if key in A:
      A[key]=A[key]+B[key]
    else:
      A[key]=B[key]
  return A

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

def count_gene(time,genes):
  for g in genes:
    if g in time:
      time[g]=time[g]+1
    else:
      time[g]=1
  return time

 
def __main__():
  description = "This script is used to filter tumor SNP/indel existed in normal sample!\n Contact with: wenyanhua@hugobiotech.com\n"
  quick_usage= 'python ' + sys.argv[0] + ' -id idir -od odir'
  newParser = argparse.ArgumentParser( description = description, usage = quick_usage );
  newParser.add_argument( "-idir", dest="idir", help="feature rank and train test dir, forced", required=True);
  newParser.add_argument( "-od", dest="od", help="output dir, forced", required=True);
  newParser.add_argument( "-n", dest="n", help="n_estimators, default sqrt(feature_num)*10");
  newParser.add_argument( "-drug", dest="drug", help="you can set multiple drugs separated by comma",required=True);
  newParser.add_argument( "-rank", dest="rank", help="feature less than rank value will be used to build model,default 100",default=100);
  newParser.add_argument( "-num", dest="num", help="the num of top features will be used to build model,if given this parameter rank parameter will be ignored");

  args = newParser.parse_args();
  if not args.idir or not args.od or not args.drug :
    newParser.print_help()
    sys.exit(1)
  global n_estimators
  args.rank=int(args.rank)
  drug=args.drug
  od=mkdir_or_die(args.od)
  idir=os.path.join(args.idir,drug)

  rank=pd.read_csv(args.idir+'/'+drug+'.genes_rank.txt',header=0,index_col=0,sep='\t')
  if not args.num:
    features=rank[rank['Mean_rank']<=args.rank].index.tolist()
  else:
    args.num=int(args.num)
    features=rank[0:args.num].index.tolist()

  if args.n:
    n_estimators=int(args.n)
  else:
    n_estimators=10*(math.ceil(math.sqrt(len(features))))
  print(n_estimators)
#  rdir=mkdir_or_die(od+'/'+drug)
  odrug=od+'/'+drug+'.acc_mcc.xls'
  fd=open(odrug,'w')
  fd.write("Drug\tRank\tACC\tMCC\tRECALL\tSpecificity\tF1\tPrecision\tME\tVME\n")
  acc=[];mcc=[];pre=[];recall=[];spe=[];f1s=[];mes=[];vmes=[];
  times=int(len(os.listdir(idir))/4)+1
#  times=10
  for i in range(1,times,1):
      X_train=pd.read_csv(idir+'/'+drug+'.train.X.'+str(i),sep="\t",header=0,index_col=0)[features]
      X_test =pd.read_csv(idir+'/'+drug+'.test.X.'+str(i),sep="\t",header=0,index_col=0)[features]
      Y_train=pd.read_csv(idir+'/'+drug+'.train.Y.'+str(i),sep="\t",header=0,index_col=0)
      Y_test =pd.read_csv(idir+'/'+drug+'.test.Y.'+str(i),sep="\t",header=0,index_col=0)
      test_pre=forestPredict(X_train,Y_train,X_test)
      me, vme = me_vme(np.array(Y_test),np.array(test_pre))
      mes.append(me);vmes.append(vme);
      a=ACC(np.array(Y_test),np.array(test_pre))
      m=MCC(np.array(Y_test),np.array(test_pre))
      precision,sensitivity,specificity,f1=cal_score(np.array(Y_test),np.array(test_pre))
      acc.append(a);mcc.append(m);pre.append(precision)
      recall.append(sensitivity);spe.append(specificity);f1s.append(f1)
      fd.write("%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n"%(drug,i,a,m,sensitivity,specificity,f1,precision,me,vme))
#      features.to_csv(rdir+'/rank.'+str(i),sep="\t",index_label="ID")
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

