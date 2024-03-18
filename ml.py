import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, accuracy_score, f1_score, confusion_matrix, precision_score, recall_score
import timeit
import psutil
import json

def reg_to_json(rmspeci, brmseti, rars_cpu, rars_gpu, rars_ram, mse, rmse, mae, r2):
    output = {
        "BATI": 0,
        "RAF1_CPU" : 0,
        "RAF1_GPU" : 0,
        "RAF1_RAM" : 0,
        "F1CI" : 0,
        "BRMSETI" : round(brmseti,2),
        "RARS_CPU" : round(rars_cpu,2),
        "RARS_GPU" : round(rars_gpu,2),
        "RARS_RAM" : round(rars_ram,2),
        "RMSPECI" : round(rmspeci,2),
        "Accuracy" : 0,
        "F1Score" : 0,
        "Precision" : 0,
        "Recall" : 0,
        "MSE" : round(mse,2),
        "RSME" : round(rmse,2),
        "MAE" : round(mae,2),
        "R2" : round(r2,2)
    }
    return json.dumps(output, indent=4)

def class_to_json(f1ci, bati, raf1_cpu, raf1_gpu, raf1_ram, accuracy, f1Score, precision, recall):
    output = {
        "BATI": round(bati,2),
        "RAF1_CPU" : round(raf1_cpu,2),
        "RAF1_GPU" : round(raf1_gpu,2),
        "RAF1_RAM" : round(raf1_ram,2),
        "F1CI" : round(f1ci,2),
        "BRMSETI" : 0,
        "RARS_CPU" : 0,
        "RARS_GPU" : 0,
        "RARS_RAM" : 0,
        "RMSPECI" : 0,
        "Accuracy" : round(accuracy,2),
        "F1Score" : round(f1Score,2),
        "Precision" : round(precision,2),
        "Recall" : round(recall,2),
        "MSE" : 0,
        "RSME" : 0,
        "MAE" : 0,
        "R2" : 0
    }
    return json.dumps(output, indent=4)

def rmspeFucntion(rmse, y_test):
  if np.mean(y_test) == 0:
    return np.nan
  else:
    return (rmse / np.mean(y_test)) * 100

def getLogReg(weight_external, weight_performance):
    #Importing Dataset
    classDataset = "https://raw.githubusercontent.com/Akshay-De-Silva/ml_apis/main/heart.csv"
    dataset = pd.read_csv(classDataset)

    #relabel values in columns to be numeric
    label_encoder = LabelEncoder()
    dataset['Sex'] = label_encoder.fit_transform(dataset['Sex'])
    dataset['ChestPainType'] = label_encoder.fit_transform(dataset['ChestPainType'])
    dataset['RestingECG'] = label_encoder.fit_transform(dataset['RestingECG'])
    dataset['ExerciseAngina'] = label_encoder.fit_transform(dataset['ExerciseAngina'])
    dataset['ST_Slope'] = label_encoder.fit_transform(dataset['ST_Slope'])

    X=dataset[['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol',
            'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope']].values
    y=dataset[['HeartDisease']].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    y_train = y_train.ravel()
    y_test = y_test.ravel()

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Logistic Regression
    from sklearn.linear_model import LogisticRegression
    logistic_regression = LogisticRegression()

    training_time = timeit.timeit(lambda: logistic_regression.fit(X_train, y_train), number=1)
    testing_time = timeit.timeit(lambda: logistic_regression.predict(X_test), number=1)

    ram_usage_bytes = psutil.virtual_memory().used
    ram_usage = ram_usage_bytes / (1024 ** 2)

    y_pred = logistic_regression.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1Score = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    max_time = 0.24
    min_time = 0.0009
    max_gpu = 2000
    min_gpu = 1000
    max_cpu = 30
    min_cpu = 15
    max_ram = 230
    min_ram = 192

    max_acc = 0.9
    min_acc = 0.81
    max_f1 =0.91
    min_f1 = 0.82

    gpu_usage = 1100
    cpu_usage = 18

    tp_cost = 1/13
    tn_cost = 1/13
    fp_cost = 4/13
    fn_cost = 7/13

    conf_matrix = confusion_matrix(y_test, y_pred)
    tp, fp, fn, tn = conf_matrix.ravel()
    f1ci = (2*tp_cost*tp)/(2*tp_cost*tp + fp_cost*fp + fn_cost*fn)

    bati = (weight_performance * ((accuracy - min_acc) / (max_acc - min_acc))) \
            + (weight_external * (1 - ((training_time - min_time) / (max_time - min_time))))
            
    raf1_cpu = (weight_performance * ((f1Score-min_f1)/(max_f1-min_f1)) + (weight_external * (1-((cpu_usage-min_cpu)/(max_cpu-min_cpu)))))
    raf1_gpu = (weight_performance * ((f1Score-min_f1)/(max_f1-min_f1)) + (weight_external * (1-((gpu_usage-min_gpu)/(max_gpu-min_gpu)))))
    raf1_ram = (weight_performance * ((f1Score-min_f1)/(max_f1-min_f1)) + (weight_external * (1-((ram_usage-min_ram)/(max_ram-min_ram)))))

    output = class_to_json(f1ci, bati, raf1_cpu, raf1_gpu, raf1_ram, accuracy, f1Score, precision, recall)
    return output

def getLinReg(weight_external, weight_performance):
    #Importing Dataset
    reg_dataset = 'https://raw.githubusercontent.com/Akshay-De-Silva/ml_apis/main/bitcoin.csv'
    dataset=pd.read_csv(reg_dataset)

    dataset = dataset.dropna()

    X = dataset[['Open', 'High', 'Low', 'Close', 'Volume_(BTC)', 'Volume_(Currency)']]
    y = dataset[['Weighted_Price']]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    y_train = np.ravel(y_train)
    y_test = np.ravel(y_test)

    # Linear Regression
    from sklearn.linear_model import LinearRegression
    linear_model = LinearRegression()

    training_time = timeit.timeit(lambda: linear_model.fit(X_train_scaled, y_train), number=1)
    testing_time = timeit.timeit(lambda: linear_model.predict(X_test_scaled), number=1)

    ram_usage_bytes = psutil.virtual_memory().used
    ram_usage = ram_usage_bytes / (1024 ** 2)

    y_pred = linear_model.predict(X_test_scaled)

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mse)
    rmspe = rmspeFucntion(rmse, y_test)

    max_time = 0.45
    min_time = 0.009
    max_gpu = 2000
    min_gpu = 1000
    max_cpu = 30
    min_cpu = 15
    max_ram = 285
    min_ram = 255

    max_rmse = 5.15
    min_rmse = 0.18
    max_r2 = 1
    min_r2 = 0.98

    gpu_usage = 1100
    cpu_usage = 18

    cost = 5000

    rmspeci = cost * rmspe

    brmseti = (weight_performance * (1-((rmse - min_rmse) / (max_rmse - min_rmse)))) \
            + (weight_external * (1 - ((training_time - min_time) / (max_time - min_time))))

    rars_cpu = (weight_performance * ((r2-min_r2)/(max_r2-min_r2)) + (weight_external * (1-((cpu_usage-min_cpu)/(max_cpu-min_cpu)))))
    rars_gpu = (weight_performance * ((r2-min_r2)/(max_r2-min_r2)) + (weight_external * (1-((gpu_usage-min_gpu)/(max_gpu-min_gpu)))))
    rars_ram = (weight_performance * ((r2-min_r2)/(max_r2-min_r2)) + (weight_external * (1-((ram_usage-min_ram)/(max_ram-min_ram)))))

    output = reg_to_json(rmspeci, brmseti, rars_cpu, rars_gpu, rars_ram, mse, rmse, mae, r2)
    return output

def getKnn(weight_external, weight_performance):
    #Importing Dataset
    classDataset = "https://raw.githubusercontent.com/Akshay-De-Silva/ml_apis/main/heart.csv"
    dataset = pd.read_csv(classDataset)

    #relabel values in columns to be numeric
    label_encoder = LabelEncoder()
    dataset['Sex'] = label_encoder.fit_transform(dataset['Sex'])
    dataset['ChestPainType'] = label_encoder.fit_transform(dataset['ChestPainType'])
    dataset['RestingECG'] = label_encoder.fit_transform(dataset['RestingECG'])
    dataset['ExerciseAngina'] = label_encoder.fit_transform(dataset['ExerciseAngina'])
    dataset['ST_Slope'] = label_encoder.fit_transform(dataset['ST_Slope'])

    X=dataset[['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol',
            'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope']].values
    y=dataset[['HeartDisease']].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    y_train = y_train.ravel()
    y_test = y_test.ravel()

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Knn
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors=3)

    training_time = timeit.timeit(lambda: knn.fit(X_train, y_train), number=1)
    testing_time = timeit.timeit(lambda: knn.predict(X_test), number=1)

    ram_usage_bytes = psutil.virtual_memory().used
    ram_usage = ram_usage_bytes / (1024 ** 2)

    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1Score = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    max_time = 0.24
    min_time = 0.0009
    max_gpu = 2000
    min_gpu = 1000
    max_cpu = 30
    min_cpu = 15
    max_ram = 230
    min_ram = 192

    max_acc = 0.9
    min_acc = 0.81
    max_f1 =0.91
    min_f1 = 0.82

    gpu_usage = 1100
    cpu_usage = 18

    tp_cost = 1/13
    tn_cost = 1/13
    fp_cost = 4/13
    fn_cost = 7/13

    conf_matrix = confusion_matrix(y_test, y_pred)
    tp, fp, fn, tn = conf_matrix.ravel()
    f1ci = (2*tp_cost*tp)/(2*tp_cost*tp + fp_cost*fp + fn_cost*fn)

    bati = (weight_performance * ((accuracy - min_acc) / (max_acc - min_acc))) \
            + (weight_external * (1 - ((training_time - min_time) / (max_time - min_time))))
            
    raf1_cpu = (weight_performance * ((f1Score-min_f1)/(max_f1-min_f1)) + (weight_external * (1-((cpu_usage-min_cpu)/(max_cpu-min_cpu)))))
    raf1_gpu = (weight_performance * ((f1Score-min_f1)/(max_f1-min_f1)) + (weight_external * (1-((gpu_usage-min_gpu)/(max_gpu-min_gpu)))))
    raf1_ram = (weight_performance * ((f1Score-min_f1)/(max_f1-min_f1)) + (weight_external * (1-((ram_usage-min_ram)/(max_ram-min_ram)))))

    output = class_to_json(f1ci, bati, raf1_cpu, raf1_gpu, raf1_ram, accuracy, f1Score, precision, recall)
    return output

def getDt(weight_external, weight_performance):
    #Importing Dataset
    classDataset = "https://raw.githubusercontent.com/Akshay-De-Silva/ml_apis/main/heart.csv"
    dataset = pd.read_csv(classDataset)

    #relabel values in columns to be numeric
    label_encoder = LabelEncoder()
    dataset['Sex'] = label_encoder.fit_transform(dataset['Sex'])
    dataset['ChestPainType'] = label_encoder.fit_transform(dataset['ChestPainType'])
    dataset['RestingECG'] = label_encoder.fit_transform(dataset['RestingECG'])
    dataset['ExerciseAngina'] = label_encoder.fit_transform(dataset['ExerciseAngina'])
    dataset['ST_Slope'] = label_encoder.fit_transform(dataset['ST_Slope'])

    X=dataset[['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol',
            'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope']].values
    y=dataset[['HeartDisease']].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    y_train = y_train.ravel()
    y_test = y_test.ravel()

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Dt
    from sklearn.tree import DecisionTreeClassifier
    dt = DecisionTreeClassifier()

    training_time = timeit.timeit(lambda: dt.fit(X_train, y_train), number=1)
    testing_time = timeit.timeit(lambda: dt.predict(X_test), number=1)

    ram_usage_bytes = psutil.virtual_memory().used
    ram_usage = ram_usage_bytes / (1024 ** 2)

    y_pred = dt.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1Score = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    max_time = 0.24
    min_time = 0.0009
    max_gpu = 2000
    min_gpu = 1000
    max_cpu = 30
    min_cpu = 15
    max_ram = 230
    min_ram = 192

    max_acc = 0.9
    min_acc = 0.81
    max_f1 =0.91
    min_f1 = 0.82

    gpu_usage = 1100
    cpu_usage = 18

    tp_cost = 1/13
    tn_cost = 1/13
    fp_cost = 4/13
    fn_cost = 7/13

    conf_matrix = confusion_matrix(y_test, y_pred)
    tp, fp, fn, tn = conf_matrix.ravel()
    f1ci = (2*tp_cost*tp)/(2*tp_cost*tp + fp_cost*fp + fn_cost*fn)

    bati = (weight_performance * ((accuracy - min_acc) / (max_acc - min_acc))) \
            + (weight_external * (1 - ((training_time - min_time) / (max_time - min_time))))
            
    raf1_cpu = (weight_performance * ((f1Score-min_f1)/(max_f1-min_f1)) + (weight_external * (1-((cpu_usage-min_cpu)/(max_cpu-min_cpu)))))
    raf1_gpu = (weight_performance * ((f1Score-min_f1)/(max_f1-min_f1)) + (weight_external * (1-((gpu_usage-min_gpu)/(max_gpu-min_gpu)))))
    raf1_ram = (weight_performance * ((f1Score-min_f1)/(max_f1-min_f1)) + (weight_external * (1-((ram_usage-min_ram)/(max_ram-min_ram)))))

    output = class_to_json(f1ci, bati, raf1_cpu, raf1_gpu, raf1_ram, accuracy, f1Score, precision, recall)
    return output

def getRf(weight_external, weight_performance):
    #Importing Dataset
    classDataset = "https://raw.githubusercontent.com/Akshay-De-Silva/ml_apis/main/heart.csv"
    dataset = pd.read_csv(classDataset)

    #relabel values in columns to be numeric
    label_encoder = LabelEncoder()
    dataset['Sex'] = label_encoder.fit_transform(dataset['Sex'])
    dataset['ChestPainType'] = label_encoder.fit_transform(dataset['ChestPainType'])
    dataset['RestingECG'] = label_encoder.fit_transform(dataset['RestingECG'])
    dataset['ExerciseAngina'] = label_encoder.fit_transform(dataset['ExerciseAngina'])
    dataset['ST_Slope'] = label_encoder.fit_transform(dataset['ST_Slope'])

    X=dataset[['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol',
            'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope']].values
    y=dataset[['HeartDisease']].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    y_train = y_train.ravel()
    y_test = y_test.ravel()

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Rf
    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(n_estimators=100)

    training_time = timeit.timeit(lambda: rf.fit(X_train, y_train), number=1)
    testing_time = timeit.timeit(lambda: rf.predict(X_test), number=1)

    ram_usage_bytes = psutil.virtual_memory().used
    ram_usage = ram_usage_bytes / (1024 ** 2)

    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1Score = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    max_time = 0.24
    min_time = 0.0009
    max_gpu = 2000
    min_gpu = 1000
    max_cpu = 30
    min_cpu = 15
    max_ram = 230
    min_ram = 192

    max_acc = 0.9
    min_acc = 0.81
    max_f1 =0.91
    min_f1 = 0.82

    gpu_usage = 1100
    cpu_usage = 18

    tp_cost = 1/13
    tn_cost = 1/13
    fp_cost = 4/13
    fn_cost = 7/13

    conf_matrix = confusion_matrix(y_test, y_pred)
    tp, fp, fn, tn = conf_matrix.ravel()
    f1ci = (2*tp_cost*tp)/(2*tp_cost*tp + fp_cost*fp + fn_cost*fn)

    bati = (weight_performance * ((accuracy - min_acc) / (max_acc - min_acc))) \
            + (weight_external * (1 - ((training_time - min_time) / (max_time - min_time))))
            
    raf1_cpu = (weight_performance * ((f1Score-min_f1)/(max_f1-min_f1)) + (weight_external * (1-((cpu_usage-min_cpu)/(max_cpu-min_cpu)))))
    raf1_gpu = (weight_performance * ((f1Score-min_f1)/(max_f1-min_f1)) + (weight_external * (1-((gpu_usage-min_gpu)/(max_gpu-min_gpu)))))
    raf1_ram = (weight_performance * ((f1Score-min_f1)/(max_f1-min_f1)) + (weight_external * (1-((ram_usage-min_ram)/(max_ram-min_ram)))))

    output = class_to_json(f1ci, bati, raf1_cpu, raf1_gpu, raf1_ram, accuracy, f1Score, precision, recall)
    return output

def getNb(weight_external, weight_performance):
    #Importing Dataset
    classDataset = "https://raw.githubusercontent.com/Akshay-De-Silva/ml_apis/main/heart.csv"
    dataset = pd.read_csv(classDataset)

    #relabel values in columns to be numeric
    label_encoder = LabelEncoder()
    dataset['Sex'] = label_encoder.fit_transform(dataset['Sex'])
    dataset['ChestPainType'] = label_encoder.fit_transform(dataset['ChestPainType'])
    dataset['RestingECG'] = label_encoder.fit_transform(dataset['RestingECG'])
    dataset['ExerciseAngina'] = label_encoder.fit_transform(dataset['ExerciseAngina'])
    dataset['ST_Slope'] = label_encoder.fit_transform(dataset['ST_Slope'])

    X=dataset[['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol',
            'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope']].values
    y=dataset[['HeartDisease']].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    y_train = y_train.ravel()
    y_test = y_test.ravel()

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Rf
    from sklearn.naive_bayes import GaussianNB
    nb = GaussianNB()

    training_time = timeit.timeit(lambda: nb.fit(X_train, y_train), number=1)
    testing_time = timeit.timeit(lambda: nb.predict(X_test), number=1)

    ram_usage_bytes = psutil.virtual_memory().used
    ram_usage = ram_usage_bytes / (1024 ** 2)

    y_pred = nb.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1Score = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    max_time = 0.24
    min_time = 0.0009
    max_gpu = 2000
    min_gpu = 1000
    max_cpu = 30
    min_cpu = 15
    max_ram = 230
    min_ram = 192

    max_acc = 0.9
    min_acc = 0.81
    max_f1 =0.91
    min_f1 = 0.82

    gpu_usage = 1100
    cpu_usage = 18

    tp_cost = 1/13
    tn_cost = 1/13
    fp_cost = 4/13
    fn_cost = 7/13

    conf_matrix = confusion_matrix(y_test, y_pred)
    tp, fp, fn, tn = conf_matrix.ravel()
    f1ci = (2*tp_cost*tp)/(2*tp_cost*tp + fp_cost*fp + fn_cost*fn)

    bati = (weight_performance * ((accuracy - min_acc) / (max_acc - min_acc))) \
            + (weight_external * (1 - ((training_time - min_time) / (max_time - min_time))))
            
    raf1_cpu = (weight_performance * ((f1Score-min_f1)/(max_f1-min_f1)) + (weight_external * (1-((cpu_usage-min_cpu)/(max_cpu-min_cpu)))))
    raf1_gpu = (weight_performance * ((f1Score-min_f1)/(max_f1-min_f1)) + (weight_external * (1-((gpu_usage-min_gpu)/(max_gpu-min_gpu)))))
    raf1_ram = (weight_performance * ((f1Score-min_f1)/(max_f1-min_f1)) + (weight_external * (1-((ram_usage-min_ram)/(max_ram-min_ram)))))

    output = class_to_json(f1ci, bati, raf1_cpu, raf1_gpu, raf1_ram, accuracy, f1Score, precision, recall)
    return output

def getSvm(weight_external, weight_performance):
    #Importing Dataset
    classDataset = "https://raw.githubusercontent.com/Akshay-De-Silva/ml_apis/main/heart.csv"
    dataset = pd.read_csv(classDataset)

    #relabel values in columns to be numeric
    label_encoder = LabelEncoder()
    dataset['Sex'] = label_encoder.fit_transform(dataset['Sex'])
    dataset['ChestPainType'] = label_encoder.fit_transform(dataset['ChestPainType'])
    dataset['RestingECG'] = label_encoder.fit_transform(dataset['RestingECG'])
    dataset['ExerciseAngina'] = label_encoder.fit_transform(dataset['ExerciseAngina'])
    dataset['ST_Slope'] = label_encoder.fit_transform(dataset['ST_Slope'])

    X=dataset[['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol',
            'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope']].values
    y=dataset[['HeartDisease']].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    y_train = y_train.ravel()
    y_test = y_test.ravel()

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Svm
    from sklearn.svm import SVC
    svm = SVC()

    training_time = timeit.timeit(lambda: svm.fit(X_train, y_train), number=1)
    testing_time = timeit.timeit(lambda: svm.predict(X_test), number=1)

    ram_usage_bytes = psutil.virtual_memory().used
    ram_usage = ram_usage_bytes / (1024 ** 2)

    y_pred = svm.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1Score = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    max_time = 0.24
    min_time = 0.0009
    max_gpu = 2000
    min_gpu = 1000
    max_cpu = 30
    min_cpu = 15
    max_ram = 230
    min_ram = 192

    max_acc = 0.9
    min_acc = 0.81
    max_f1 =0.91
    min_f1 = 0.82

    gpu_usage = 1100
    cpu_usage = 18

    tp_cost = 1/13
    tn_cost = 1/13
    fp_cost = 4/13
    fn_cost = 7/13

    conf_matrix = confusion_matrix(y_test, y_pred)
    tp, fp, fn, tn = conf_matrix.ravel()
    f1ci = (2*tp_cost*tp)/(2*tp_cost*tp + fp_cost*fp + fn_cost*fn)

    bati = (weight_performance * ((accuracy - min_acc) / (max_acc - min_acc))) \
            + (weight_external * (1 - ((training_time - min_time) / (max_time - min_time))))
            
    raf1_cpu = (weight_performance * ((f1Score-min_f1)/(max_f1-min_f1)) + (weight_external * (1-((cpu_usage-min_cpu)/(max_cpu-min_cpu)))))
    raf1_gpu = (weight_performance * ((f1Score-min_f1)/(max_f1-min_f1)) + (weight_external * (1-((gpu_usage-min_gpu)/(max_gpu-min_gpu)))))
    raf1_ram = (weight_performance * ((f1Score-min_f1)/(max_f1-min_f1)) + (weight_external * (1-((ram_usage-min_ram)/(max_ram-min_ram)))))

    output = class_to_json(f1ci, bati, raf1_cpu, raf1_gpu, raf1_ram, accuracy, f1Score, precision, recall)
    return output

def getRr(weight_external, weight_performance):
    #Importing Dataset
    reg_dataset = 'https://raw.githubusercontent.com/Akshay-De-Silva/ml_apis/main/bitcoin.csv'
    dataset=pd.read_csv(reg_dataset)

    dataset = dataset.dropna()

    X = dataset[['Open', 'High', 'Low', 'Close', 'Volume_(BTC)', 'Volume_(Currency)']]
    y = dataset[['Weighted_Price']]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    y_train = np.ravel(y_train)
    y_test = np.ravel(y_test)

    # Ridge Regression
    from sklearn.linear_model import Ridge
    rr = Ridge()

    training_time = timeit.timeit(lambda: rr.fit(X_train_scaled, y_train), number=1)
    testing_time = timeit.timeit(lambda: rr.predict(X_test_scaled), number=1)

    ram_usage_bytes = psutil.virtual_memory().used
    ram_usage = ram_usage_bytes / (1024 ** 2)

    y_pred = rr.predict(X_test_scaled)

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mse)
    rmspe = rmspeFucntion(rmse, y_test)

    max_time = 0.45
    min_time = 0.009
    max_gpu = 2000
    min_gpu = 1000
    max_cpu = 30
    min_cpu = 15
    max_ram = 285
    min_ram = 255

    max_rmse = 5.15
    min_rmse = 0.18
    max_r2 = 1
    min_r2 = 0.98

    gpu_usage = 1100
    cpu_usage = 18

    cost = 5000

    rmspeci = cost * rmspe

    brmseti = (weight_performance * (1-((rmse - min_rmse) / (max_rmse - min_rmse)))) \
            + (weight_external * (1 - ((training_time - min_time) / (max_time - min_time))))

    rars_cpu = (weight_performance * ((r2-min_r2)/(max_r2-min_r2)) + (weight_external * (1-((cpu_usage-min_cpu)/(max_cpu-min_cpu)))))
    rars_gpu = (weight_performance * ((r2-min_r2)/(max_r2-min_r2)) + (weight_external * (1-((gpu_usage-min_gpu)/(max_gpu-min_gpu)))))
    rars_ram = (weight_performance * ((r2-min_r2)/(max_r2-min_r2)) + (weight_external * (1-((ram_usage-min_ram)/(max_ram-min_ram)))))

    output = reg_to_json(rmspeci, brmseti, rars_cpu, rars_gpu, rars_ram, mse, rmse, mae, r2)
    return output

def getLasso(weight_external, weight_performance):
    #Importing Dataset
    reg_dataset = 'https://raw.githubusercontent.com/Akshay-De-Silva/ml_apis/main/bitcoin.csv'
    dataset=pd.read_csv(reg_dataset)

    dataset = dataset.dropna()

    X = dataset[['Open', 'High', 'Low', 'Close', 'Volume_(BTC)', 'Volume_(Currency)']]
    y = dataset[['Weighted_Price']]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    y_train = np.ravel(y_train)
    y_test = np.ravel(y_test)

    # Lasso Regression
    from sklearn.linear_model import Lasso
    lasso = Lasso()

    training_time = timeit.timeit(lambda: lasso.fit(X_train_scaled, y_train), number=1)
    testing_time = timeit.timeit(lambda: lasso.predict(X_test_scaled), number=1)

    ram_usage_bytes = psutil.virtual_memory().used
    ram_usage = ram_usage_bytes / (1024 ** 2)

    y_pred = lasso.predict(X_test_scaled)

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mse)
    rmspe = rmspeFucntion(rmse, y_test)

    max_time = 0.45
    min_time = 0.009
    max_gpu = 2000
    min_gpu = 1000
    max_cpu = 30
    min_cpu = 15
    max_ram = 285
    min_ram = 255

    max_rmse = 5.15
    min_rmse = 0.18
    max_r2 = 1
    min_r2 = 0.98

    gpu_usage = 1100
    cpu_usage = 18

    cost = 5000

    rmspeci = cost * rmspe

    brmseti = (weight_performance * (1-((rmse - min_rmse) / (max_rmse - min_rmse)))) \
            + (weight_external * (1 - ((training_time - min_time) / (max_time - min_time))))

    rars_cpu = (weight_performance * ((r2-min_r2)/(max_r2-min_r2)) + (weight_external * (1-((cpu_usage-min_cpu)/(max_cpu-min_cpu)))))
    rars_gpu = (weight_performance * ((r2-min_r2)/(max_r2-min_r2)) + (weight_external * (1-((gpu_usage-min_gpu)/(max_gpu-min_gpu)))))
    rars_ram = (weight_performance * ((r2-min_r2)/(max_r2-min_r2)) + (weight_external * (1-((ram_usage-min_ram)/(max_ram-min_ram)))))

    output = reg_to_json(rmspeci, brmseti, rars_cpu, rars_gpu, rars_ram, mse, rmse, mae, r2)
    return output

def getBrr(weight_external, weight_performance):
    #Importing Dataset
    reg_dataset = 'https://raw.githubusercontent.com/Akshay-De-Silva/ml_apis/main/bitcoin.csv'
    dataset=pd.read_csv(reg_dataset)

    dataset = dataset.dropna()

    X = dataset[['Open', 'High', 'Low', 'Close', 'Volume_(BTC)', 'Volume_(Currency)']]
    y = dataset[['Weighted_Price']]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    y_train = np.ravel(y_train)
    y_test = np.ravel(y_test)

    # Bayesian Ridge Regression
    from sklearn.linear_model import BayesianRidge
    brr = BayesianRidge()

    training_time = timeit.timeit(lambda: brr.fit(X_train_scaled, y_train), number=1)
    testing_time = timeit.timeit(lambda: brr.predict(X_test_scaled), number=1)

    ram_usage_bytes = psutil.virtual_memory().used
    ram_usage = ram_usage_bytes / (1024 ** 2)

    y_pred = brr.predict(X_test_scaled)

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mse)
    rmspe = rmspeFucntion(rmse, y_test)

    max_time = 0.45
    min_time = 0.009
    max_gpu = 2000
    min_gpu = 1000
    max_cpu = 30
    min_cpu = 15
    max_ram = 285
    min_ram = 255

    max_rmse = 5.15
    min_rmse = 0.18
    max_r2 = 1
    min_r2 = 0.98

    gpu_usage = 1100
    cpu_usage = 18

    cost = 5000

    rmspeci = cost * rmspe

    brmseti = (weight_performance * (1-((rmse - min_rmse) / (max_rmse - min_rmse)))) \
            + (weight_external * (1 - ((training_time - min_time) / (max_time - min_time))))

    rars_cpu = (weight_performance * ((r2-min_r2)/(max_r2-min_r2)) + (weight_external * (1-((cpu_usage-min_cpu)/(max_cpu-min_cpu)))))
    rars_gpu = (weight_performance * ((r2-min_r2)/(max_r2-min_r2)) + (weight_external * (1-((gpu_usage-min_gpu)/(max_gpu-min_gpu)))))
    rars_ram = (weight_performance * ((r2-min_r2)/(max_r2-min_r2)) + (weight_external * (1-((ram_usage-min_ram)/(max_ram-min_ram)))))

    output = reg_to_json(rmspeci, brmseti, rars_cpu, rars_gpu, rars_ram, mse, rmse, mae, r2)
    return output

def getEn(weight_external, weight_performance):
    #Importing Dataset
    reg_dataset = 'https://raw.githubusercontent.com/Akshay-De-Silva/ml_apis/main/bitcoin.csv'
    dataset=pd.read_csv(reg_dataset)

    dataset = dataset.dropna()

    X = dataset[['Open', 'High', 'Low', 'Close', 'Volume_(BTC)', 'Volume_(Currency)']]
    y = dataset[['Weighted_Price']]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    y_train = np.ravel(y_train)
    y_test = np.ravel(y_test)

    # Elastic Net Regression
    from sklearn.linear_model import ElasticNet
    en = ElasticNet()

    training_time = timeit.timeit(lambda: en.fit(X_train_scaled, y_train), number=1)
    testing_time = timeit.timeit(lambda: en.predict(X_test_scaled), number=1)

    ram_usage_bytes = psutil.virtual_memory().used
    ram_usage = ram_usage_bytes / (1024 ** 2)

    y_pred = en.predict(X_test_scaled)

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mse)
    rmspe = rmspeFucntion(rmse, y_test)

    max_time = 0.45
    min_time = 0.009
    max_gpu = 2000
    min_gpu = 1000
    max_cpu = 30
    min_cpu = 15
    max_ram = 285
    min_ram = 255

    max_rmse = 5.15
    min_rmse = 0.18
    max_r2 = 1
    min_r2 = 0.98

    gpu_usage = 1100
    cpu_usage = 18

    cost = 5000

    rmspeci = cost * rmspe

    brmseti = (weight_performance * (1-((rmse - min_rmse) / (max_rmse - min_rmse)))) \
            + (weight_external * (1 - ((training_time - min_time) / (max_time - min_time))))

    rars_cpu = (weight_performance * ((r2-min_r2)/(max_r2-min_r2)) + (weight_external * (1-((cpu_usage-min_cpu)/(max_cpu-min_cpu)))))
    rars_gpu = (weight_performance * ((r2-min_r2)/(max_r2-min_r2)) + (weight_external * (1-((gpu_usage-min_gpu)/(max_gpu-min_gpu)))))
    rars_ram = (weight_performance * ((r2-min_r2)/(max_r2-min_r2)) + (weight_external * (1-((ram_usage-min_ram)/(max_ram-min_ram)))))

    output = reg_to_json(rmspeci, brmseti, rars_cpu, rars_gpu, rars_ram, mse, rmse, mae, r2)
    return output
