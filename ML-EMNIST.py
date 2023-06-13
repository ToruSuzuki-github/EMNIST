from sys import stdout,argv
from ml_algorithm import kNN_EMNIST,SVM_EMNIST

if __name__=="__main__":
    algorithm_set={"kNN","SVM"}
    run_flag=False
    if len(argv)==2:
        if argv[1] in {"-help","-h"}:
            stdout.write("\nUsage of Program Files")
            stdout.write("\nPredicts EMNIST class by machine learning rhythm.")
            stdout.write("\nIf you want to know more about the details, please do a.")
            for algorithm in list(algorithm_set):
                stdout.write("\n - python3 "+str(argv[0])+" "+str(algorithm)+" <-h,-help>")
            stdout.write("\n")
            stdout.flush()
    elif len(argv)>2:
        if argv[1] in algorithm_set:
            if argv[1]=="kNN":
                argv[0]=argv[0]+" "+argv[1]
                argv.pop(1)
                kNN_EMNIST.knn_emnist(argv)
            elif argv[1]=="SVM":
                argv[0]=argv[0]+" "+argv[1]
                argv.pop(1)
                SVM_EMNIST.svm_emnist(argv)
