#!/share/data1/wenyanhua/soft/current/Rscript

library(getopt)
spec=matrix(c(
	'help','h',0,"logical",
	'idir','i',1,"character",
	'odir','o',1,"character",
	'drug','d',1,"character"
	),byrow=TRUE,ncol=4
)
opt=getopt(spec)
print_usage<-function(spec=NULL){
	cat(getopt(spec,usage=TRUE));
	cat("Usage example: \n")
	cat("
		Description: 
		Contact: Yanhua Wen <wenyanhua@hugobiotech.com>	
		Usage example:
		Options:
		--help		NULL		get this help
		--idir	-i	input dir
		--odir	-o	output dir
		--drug	-d	durg default all
		\n")

	q(status=1);
}
library(caret)
calculate<-function(real,prob){
	pred<-ifelse(prob>=0.5,1,0)
	con=confusionMatrix(factor(pred),factor(real))
	acc=con$overall[1]
	vme	=1-con$byClass["Sensitivity"]	##Sensitivity
	me	=1-con$byClass["Specificity"]	##Specificity
	f1	=con$byClass["F1"]
	mc	=ModelMetrics::mcc(real,prob,0.5)
	names(mc)="MCC"
	if(is.na(mc))	mc=0
	if(is.na(f1))	f1=0
	return(c(acc,mc,f1,me,vme))
}
if(is.null(opt$idir)) print_usage(spec)
if(is.null(opt$odir)) print_usage(spec)
dir.create(opt$odir,showWarnings=F)

sir<-c(0,1)
names(sir)<-c("S","R")

dirs<-list.dirs(opt$idir,full.names=T)
dirs<-dirs[-c(1,grep("rank",dirs))]

if(!is.null(opt$drug)){
	drugs<-strsplit(opt$drug,split=",")[[1]]
	dirs<- dirs[basename(dirs) %in% drugs]
}
library(e1071)
library(pROC)
##Drug    Rank    ACC     MCC     RECALL  Specificity     F1      Precision       ME      VME
for(dir in dirs){
	drug=basename(dir)
	rankf<-paste0(dir,".genes_rank.txt")
	rank=read.csv(rankf,header=T,sep="\t",row.names=1)
	genes<-rownames(rank)[rank[,1]<=100]
	trainxf<-list.files(dir,full.names=T,pattern="train.X")
	trainyf<-list.files(dir,full.names=T,pattern="train.Y")
	testxf<-list.files(dir,full.names=T,pattern="test.X")
	testyf<-list.files(dir,full.names=T,pattern="test.Y")
	write.table(t(c("Drug","Ranks","ACC","MCC","F1","ME","VME")),paste0(opt$odir,"/",drug,".svm.xls"),row.names=F,col.names=F,sep="\t",quote=F)
	write.table(t(c("Drug","Ranks","ACC","MCC","F1","ME","VME")),paste0(opt$odir,"/",drug,".logistic.xls"),row.names=F,col.names=F,sep="\t",quote=F)
	ma1<-matrix(nr=length(trainxf),nc=6)
	ma2<-matrix(nr=length(trainxf),nc=6)

	for(i in 1:length(trainxf)){
		base=as.numeric(strsplit(basename(trainxf[i]),split="[.]")[[1]][4])
		trainx<-read.csv(trainxf[i],header=T,sep="\t",row.names=1,check.names=F)[,genes]
		trainy<-read.csv(trainyf[i],header=T,sep="\t",row.names=1,check.names=F)
		train<-cbind(sir[trainy[,1]],trainx)
		colnames(train)[1]<-"AMR"
		testx<-read.csv(testxf[i],header=T,sep="\t",row.names=1,check.names=F)[,genes]
		testy<-read.csv(testyf[i],header=T,sep="\t",row.names=1,check.names=F)
		test<-cbind(sir[testy[,1]],testx)
		colnames(test)[1]<-"AMR"

		###svm
		model1=svm(AMR~.,data=train)
		prob1<-predict(model1,newdata=test)
		value1<-calculate(test$AMR,prob1)
		ma1[i,]<-c(base,value1)
		##logistic
		model2<-glm(AMR~.,data=train,family=binomial("logit"),maxit=100)
		prob2<-predict(model2,newdata=test)
		value2<-calculate(test$AMR,prob2)
		ma2[i,]<-c(base,value2)
	}
	write.table(cbind(drug,ma1),paste0(opt$odir,"/",drug,".svm.xls"),row.names=F,col.names=F,sep="\t",quote=F,append=T)
	write.table(t(c(drug,"Mean",colMeans(ma1)[-1])),paste0(opt$odir,"/",drug,".svm.xls"),row.names=F,col.names=F,sep="\t",quote=F,append=T)
	write.table(cbind(drug,ma2),paste0(opt$odir,"/",drug,".logistic.xls"),row.names=F,col.names=F,sep="\t",quote=F,append=T)
	write.table(t(c(drug,"Mean",colMeans(ma2)[-1])),paste0(opt$odir,"/",drug,".logistic.xls"),row.names=F,col.names=F,sep="\t",quote=F,append=T)
}

