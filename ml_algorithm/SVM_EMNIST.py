from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
import pandas as pd
from sys import stdout,argv
import os
import matplotlib.pyplot as plt
import csv
import time
from copy import copy

def SupportVectorMachine(dataset_file_path,kernel_kind,c_start,c_end,seed):
    #------------パラメーター------------
    #出力ディレクトリパス
    output_directory_path=dataset_file_path.split("/")
    output_directory_path[-4]="SVM"
    output_directory_path="/".join(output_directory_path)
    os.makedirs(output_directory_path,exist_ok=True)

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

    #------------SVM実行------------
    stdout.write("\nrun SVM now")
    stdout.flush()
    #出力用辞書
    result_dict={}
    for regularization in range(c_start,c_end+1,1):
        #アルゴリズム設定(参照:)
        algorithm=SVC(C=regularization # 正則化パラメーター,自然数,デフォルト=1
                    ,kernel=kernel_kind # カーネルの種類,{"rbf":ガウスカーネル,"linear":線形,"poly":多項式,"sigmoid":シグモイド関数},デフォルト="rbf"
                    #,degree= # 多項式カーネルの次数,自然数,デフォルト=3
                    #,gamma= # ガウスカーネルとシグモイドカーネルの係数,{"scale":1/(特徴次元数 * データセットの分散),"auto":1/特徴次元数},デフォルト="scale"
                    #,probability= #予測時に各クラスに属する確率を返す,真偽値,デフォルト=False
                    #cache_size# # キャッシュサイズ（MB）,数値,デフォルト=200
                    ,random_state=seed #乱数のシード,{整数:シードの指定,None:ランダムシード},デフォルト=None
        )
        #モデル構築（学習）
        stdout.write("\n c="+str(regularization)+" train now")
        stdout.flush()
        start_training_time=time.time()
        algorithm.fit(training_dataset,training_labels)
        stdout.write("completed")
        stdout.flush()
        #テスト実行
        stdout.write("\n c="+str(regularization)+" test now")
        stdout.flush()
        start_test_time=time.time()
        predict_result=algorithm.predict(test_dataset)
        end_test_time=time.time()
        stdout.write("completed")
        stdout.flush()
        #評価値算出
        result_dict[regularization]={"accuracy":accuracy_score(test_labels,predict_result)
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
        output_predict_csv_path=output_directory_path+"/"+str(kernel_kind)+"_"+str(regularization)+"_"+str(seed)+"_predict.csv"
        result.to_csv(output_predict_csv_path)
        stdout.write("\noutput predict csv path: "+str(output_predict_csv_path))
        stdout.flush()
    stdout.write("\nrun SVM completed")
    stdout.flush()

    #------------精度と処理時間のファイル出力------------
    stdout.write("\noutputing accuracy and run time now")
    stdout.flush()
    #評価指標による評価結果の画像出力
    for valuation_index in ["accuracy","precision_score","recall_score","f1_score"]:
        plt.figure()
        plt.xlabel("c (regularization-parameter)")
        plt.ylabel(str(valuation_index))
        plt.plot(list(result_dict.keys()),[values[valuation_index] for values in result_dict.values()])
        output_valuation_image_path=output_directory_path+"/"+str(kernel_kind)+"_"+str(c_start)+"_"+str(c_end)+"_"+str(seed)+"_"+str(valuation_index)+".png"
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
    plt.title("SVM run time transition")
    handler1, label1 = ax1.get_legend_handles_labels()
    handler2, label2 = ax2.get_legend_handles_labels()
    ax1.legend(handler1 + handler2, label1 + label2, loc="best", borderaxespad=0)
    ax1.set_xlabel("c (regularization-parameter)")
    ax1.set_ylabel("training time")
    ax2.set_ylabel("test time")
    output_run_time_image_path=output_directory_path+"/"+str(kernel_kind)+"_"+str(c_start)+"_"+str(c_end)+"_"+str(seed)+"_rum-time.png"
    if os.path.isfile(output_run_time_image_path):
        os.remove(output_run_time_image_path)
    plt.savefig(output_run_time_image_path)
    plt.clf
    #精度と実行時間の推移の表を出力
    output_transition_csv_path=output_directory_path+"/"+str(kernel_kind)+"_"+str(c_start)+"_"+str(c_end)+"_"+str(seed)+"_transition.csv"
    with open(output_transition_csv_path,"w") as wf:
        csv_writer=csv.writer(wf)
        csv_writer.writerow(["c","accuracy","training_time","test_time"])
        csv_writer.writerows([[c,values["accuracy"],values["training_time"],values["test_time"]] for c,values in result_dict.items()])
    stdout.write("\noutputing accuracy and run time completed")

    #出力ファイルパスを標準出力
    stdout.write("\noutput valuation image path: "+str(output_valuation_image_path.rsplit("_",1)[0]+"_<valuation index>.png"))
    stdout.write("\noutput run time image path: "+str(output_run_time_image_path))
    stdout.write("\noutput transition csv path: "+str(output_transition_csv_path))
    stdout.flush()

    return True

def svm_emnist(argv):
    run_flag=False
    kernel_kind_set={"rbf","linear","poly","sigmoid"}
    if argv[1] in {"-help","-h"}:
        stdout.write("\nUsage of Program Files")
        stdout.write("\nProgram file to apply SVM (Support Vector Machine) to the specified data set (features).")
        stdout.write("\nMultiple consecutive c-values can be run simultaneously.")
        stdout.write("\nDetails of command line arguments")
        stdout.write("\nPython3 "+str(argv[0])+" "\
                     "<File path of the dataset csv file. ex:./EMNIST/feature/mesh/nomal/240000_mesh_3x3.csv> "\
                        "<Kernel kind. rbf,linear,poly,sigmoid> "\
                            "<Starting value of c-value. Natural number.> "\
                                "<End value of c-value. Natural number.> "\
                                    "<Seeding of random numbers. Integer or None>"
                                    )
        stdout.flush()
    elif os.path.isfile(argv[1]):
        dataset_file_path=argv[1]
        if len(argv)==6:
            if argv[2] in kernel_kind_set:
                kernel_kind=argv[2]
                try:
                    c_start=int(argv[3])
                    c_end=int(argv[4])
                    if argv[5]=="None":
                        seed=None
                    else:
                        seed=int(argv[5])
                    run_flag=True
                except ValueError as e:
                    raise(e)
        else:
            stdout.write("\nInvalid number of command line arguments.")
    else:
        stdout.write("\nFile path is invalid.")
    #SVMの実行と結果処理
    result=False
    if run_flag:
        result=SupportVectorMachine(dataset_file_path,kernel_kind,c_start,c_end,seed)
    if result:
        stdout.write("\nNormal termination")
    else:
        stdout.write("\nAbnormal Termination")
    stdout.write("\n")
    stdout.flush()

if __name__=="__main__":
    svm_emnist(argv)