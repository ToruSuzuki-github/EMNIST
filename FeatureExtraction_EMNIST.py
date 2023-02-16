from sys import stdout,argv
from feature_extraction import ExtractionPixelValueFeature_EMNIST
from feature_extraction import ExtractionMeshFeature_EMNIST

#------------パラメーター------------
unfinished_flag=False
unfinished_data_num=1
binary_threshold=128
if unfinished_flag:
    stdout.write('\nTest run mode')
    stdout.flush()


stdout.write('\nStart '+" ".join(argv))
stdout.flush()
result=False
if argv[1]=='help':
    stdout.write('\n<<Usage of Program Files>>')
    stdout.write('\nProgram to extract features from EMNIST digits.')
    stdout.write('\n<<Details of command line arguments>>')
    stdout.write("\Pixel Value Features: <'pixel_value'> <Binarization or not（True or False）>")
    stdout.write("\nMesh Features: <'mesh'> <Number of divisions in x-axis direction> <Number of divisions along y-axis> <Binarization or not（True or False）> <Whether to delete marginal rows and columns.（True or False）>")
#ピクセル値特徴量
elif argv[1]=='pixel_value':
    if len(argv)==3 and argv[2] in {'True','False'}:
        if argv[2]=='True':
            binarization_flag=True
        else:
            binarization_flag=False
        result=ExtractionPixelValueFeature_EMNIST.featureExtraction(unfinished_flag,unfinished_data_num,binary_threshold,binarization_flag)
    else:
        stdout.write('\nThe second and subsequent command line arguments are invalid.')
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
        stdout.write('\nThe second and subsequent command line arguments are invalid.')
else:
    stdout.write('\nThe first argument of the command line argument is invalid')
if result:
    stdout.write('\nNormal termination')
else:
    stdout.write('\nAbnormal Termination')
stdout.write('\nCompleted '+" ".join(argv))
stdout.write('\n')
stdout.flush()