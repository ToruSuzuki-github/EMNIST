from sys import stdout,argv
from feature_extraction import ExtractionPixelValueFeature_EMNIST
from feature_extraction import ExtractionMeshFeature_EMNIST

#------------パラメーター------------
unfinished_flag=False
unfinished_data_num=1
binary_threshold=128
if unfinished_flag:
    stdout.write('\n試験実行モード')
    stdout.flush()


stdout.write('\n実行開始 '+" ".join(argv))
stdout.flush()
result=False
if argv[1]=='help':
    stdout.write('\n【プログラムの用途】')
    stdout.write('\nEMNISTのdigitsから特徴量を抽出するプログラム')
    stdout.write('\n【コマンドライン引数】')
    stdout.write("\nピクセル値特徴量: <'pixel_value'> <二値化するかどうか（True or False）>")
    stdout.write("\nメッシュ特徴量: <'mesh'> <x軸方向の分割数> <y軸方向の分割数> <二値化するかどうか（True or False）> <余白行、列を削除するかどうか（True or False）>")
#ピクセル値特徴量
elif argv[1]=='pixel_value':
    if len(argv)==3 and argv[2] in {'True','False'}:
        if argv[2]=='True':
            binarization_flag=True
        else:
            binarization_flag=False
        result=ExtractionPixelValueFeature_EMNIST.featureExtraction(unfinished_flag,unfinished_data_num,binary_threshold,binarization_flag)
    else:
        stdout.write('\nコマンドライン引数の第二引数以降が不正')
#メッシュ特徴量
elif argv[1]=='mesh':
    if len(argv)==6 and argv[4] in {'True','False'} and argv[5] in {'True','False'}:
        if argv[4]=='True':
            binarization_flag=True
        else:
            binarization_flag=False
        if argv[5]=='True':
            remove_surplus_flag=True
        else:
            remove_surplus_flag=False
        try:
            mesh_x_num,mesh_y_num=int(argv[2]),int(argv[3])
            if mesh_x_num>0 and mesh_y_num>0:
                result=ExtractionMeshFeature_EMNIST.featureExtraction(mesh_x_num,mesh_y_num,unfinished_flag,unfinished_data_num,binary_threshold,binarization_flag,remove_surplus_flag)
        except ValueError as e:
            raise(e)
    else:
        stdout.write('\nコマンドライン引数の第二引数以降が不正')
else:
    stdout.write('\nコマンドライン引数の第一引数が不正')
if result:
    stdout.write('\n正常終了')
else:
    stdout.write('\n異常終了')
stdout.write('\n実行終了 '+" ".join(argv))
stdout.write('\n')
stdout.flush()