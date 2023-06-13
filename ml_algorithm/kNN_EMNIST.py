from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
import pandas as pd
from sys import stdout,argv
import os
import matplotlib.pyplot as plt
import csv
import time
from copy import copy

def kNearestNeighbor(dataset_file_path,k_start,k_end):
    #------------パラメーター------------
    #出力ディレクトリパス
    output_directory_path=dataset_file_path.split("/")
    output_directory_path[-4]="kNN"
    output_directory_path="/".join(output_directory_path)
    os.makedirs(output_directory_path,exist_ok=True)
    #kNNにおける各データの重み
    weight="uniform"# 比較する学習データの重み,{"uniform":均等,"distance":学習データとの距離の逆数,ユーザー定義の関数,None},デフォルト="uniform

    #------------データセット読み込み------------
    #データセット読み込み
    stdout.write("\nread csv now")
    stdout.flush
    dataset=pd.read_csv(dataset_file_path,index_col=0)
    #データセットの分割
    training_dataset_amount=int(os.path.basename(dataset_file_path).split("_",1)[0])
    dataset_columns_name=dataset.columns.to_list()
    dataset_columns_name.remove("labels")
    training_dataset=dataset.loc[:training_dataset_amount-1,dataset_columns_name]
    training_labels=dataset.loc[:training_dataset_amount-1,"labels"]
    test_dataset=dataset.loc[training_dataset_amount:,dataset_columns_name]
    test_labels=dataset.loc[training_dataset_amount:,"labels"]
    stdout.write("\nread csv completed")
    stdout.flush

    #------------kNN実行------------
    stdout.write("\nrun kNN now")
    stdout.flush()
    #出力用辞書
    result_dict={}
    for k in range(k_start,k_end+1,1):
        #アルゴリズム設定(参照:https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)
        algorithm=KNeighborsClassifier(n_neighbors=k # 比較する学習データの数,整数,デフォルト=5
                                ,weights=weight # 比較する学習データの重み,{"uniform":均等,"distance":学習データとの距離の逆数,ユーザー定義の関数,None},デフォルト="uniform
                                ,algorithm="auto" #最近傍計算に使用,{"auto":自動で決定した最適なアルゴリズム,"ball_tree":BallTree,"kd_tree":KDTree,"brute":総当たり探索},デフォルト="auto"
                                #,leaf_size=30 #BallTreeやKDTreeに渡す葉の数,整数,デフォルト=30
                                #,p #ミルコフスキー軽量の検出力パラメーター,整数,デフォルト=2
                                ,metric="minkowski" #距離計算法,{文字列,呼び出し可能なもの},デフォルト="minkowski"
                                #,metric_params=None #metricに与える引数,{辞書,None},デフォルト=None
                                #,n_jobs=None #近隣探索の際に並列実行するジョブ数,{整数,None},デフォルト=None
        )
        #モデル構築（学習）
        stdout.write("\n k="+str(k)+" train now")
        stdout.flush()
        start_training_time=time.time()
        algorithm.fit(training_dataset,training_labels)
        stdout.write("completed")
        stdout.flush()
        #テスト実行
        stdout.write("\n k="+str(k)+" test now")
        stdout.flush()
        start_test_time=time.time()
        predict_result=algorithm.predict(test_dataset)
        end_test_time=time.time()
        stdout.write("completed")
        stdout.flush()
        #評価値算出
        result_dict[k]={"accuracy":accuracy_score(test_labels,predict_result)
                        ,"precision_score":precision_score(test_labels,predict_result)
                        ,"recall_score":recall_score(test_labels,predict_result)
                        ,"f1_score":f1_score(test_labels,predict_result)
                        ,"training_time":start_test_time-start_training_time
                        ,"test_time":end_test_time-start_test_time
                        }
        #処理時間を標準出力
        stdout.write(" ("+str(end_test_time-start_training_time)+"s)")
        #predict結果の詳細をファイル出力
        result=pd.DataFrame(test_labels)
        result["predict"]=predict_result
        output_predict_csv_path=output_directory_path+"/"+str(weight)+"_"+str(k)+"_predict.csv"
        result.to_csv(output_predict_csv_path)
        stdout.write("\noutput predict csv path: "+str(output_predict_csv_path))
        stdout.flush()
    stdout.write("\nrun kNN completed")
    stdout.flush()

    #------------精度と処理時間のファイル出力------------
    stdout.write("\noutputing accuracy and run time now")
    stdout.flush()
    #評価指標による評価結果の画像出力
    for valuation_index in ["accuracy","precision_score","recall_score","f1_score"]:
        plt.figure()
        plt.xlabel("k (n_neighbors)")
        plt.ylabel(str(valuation_index))
        plt.plot(list(result_dict.keys()),[values[valuation_index] for values in result_dict.values()])
        output_valuation_image_path=output_directory_path+"/"+str(weight)+"_"+str(k_start)+"_"+str(k_end)+"_"+str(valuation_index)+".png"
        if os.path.isfile(output_valuation_image_path):
            os.remove(output_valuation_image_path)
        plt.savefig(output_valuation_image_path)
        plt.clf
    #実行時間の画像出力
    fig, ax1=plt.subplots()
    ax2=ax1.twinx()
    ax1.plot(list(result_dict.keys())
            ,[values["training_time"] for values in result_dict.values()]
            ,color="r"
            ,label="training_time"
            )
    ax2.plot(list(result_dict.keys())
            ,[values["test_time"] for values in result_dict.values()]
            ,color="b"
            ,label="test_time"
            )
    plt.title("kNN run time transition")
    handler1, label1 = ax1.get_legend_handles_labels()
    handler2, label2 = ax2.get_legend_handles_labels()
    ax1.legend(handler1 + handler2, label1 + label2, loc="best", borderaxespad=0)
    ax1.set_xlabel("k (n_neighbors)")
    ax1.set_ylabel("training time")
    ax2.set_ylabel("test time")
    output_run_time_image_path=output_directory_path+"/"+str(weight)+"_"+str(k_start)+"_"+str(k_end)+"rum_time.png"
    if os.path.isfile(output_run_time_image_path):
        os.remove(output_run_time_image_path)
    plt.savefig(output_run_time_image_path)
    plt.clf
    #精度と実行時間の推移の表を出力
    output_transition_csv_path=output_directory_path+"/"+str(weight)+"_"+str(k_start)+"_"+str(k_end)+"_transition.csv"
    with open(output_transition_csv_path,"w") as wf:
        csv_writer=csv.writer(wf)
        csv_writer.writerow(["k","accuracy","training_time","test_time"])
        csv_writer.writerows([[k,values["accuracy"],values["training_time"],values["test_time"]] for k,values in result_dict.items()])
    stdout.write("\noutputing accuracy and run time completed")

    #出力ファイルパスを標準出力
    stdout.write("\noutput valuation image path: "+str(output_valuation_image_path.rsplit("_",1)[0]+"_<valuation index>.png"))
    stdout.write("\noutput run time image path: "+str(output_run_time_image_path))
    stdout.write("\noutput transition csv path: "+str(output_transition_csv_path))
    stdout.flush()

    return True

def knn_emnist(argv):
    run_flag=False
    if argv[1] in {"-help","-h"}:
        stdout.write("\nUsage of Program Files")
        stdout.write("\nProgram file to apply k Nearest Neighbor to the specified data set (features).")
        stdout.write("\nMultiple consecutive k-values can be run simultaneously.")
        stdout.write("\nDetails of command line arguments")
        stdout.write("\npython3 "+str(argv[0])+" "\
                     "<File path of the dataset csv file. ex:./EMNIST/feature/mesh/nomal/240000_mesh_3x3.csv> "\
                        "<Starting value of k-value. Natural number.> "\
                            "<End value of k-value. Natural number.>"
                            )
        stdout.flush()
    elif os.path.isfile(argv[1]):
        dataset_file_path=argv[1]
        if len(argv)==3:
            try:
                k_start=k_end=int(argv[2])
                run_flag=True
            except ValueError as e:
                raise(e)
        elif len(argv)==4:
            try:
                k_start=int(argv[2])
                k_end=int(argv[3])
                run_flag=True
            except ValueError as e:
                raise(e)
        else:
            stdout.write("\nInvalid number of command line arguments.")
    else:
        stdout.write("\nFile path is invalid.")
    #kNNの実行と結果処理
    result=False
    if run_flag:
        result=kNearestNeighbor(dataset_file_path,k_start,k_end)
    if result:
        stdout.write("\nNormal termination")
    else:
        stdout.write("\nAbnormal Termination")
    stdout.write("\n")
    stdout.flush()

if __name__=="__main__":
    knn_emnist(argv)