# download img file

# split train / val 8:2

# move file to train / val directory


import boto3
import json
import os
import pandas as pd
import requests
from tqdm import tqdm
import PIL.Image as Image

def objects_processing(row):
    objects = row.objects
    objects = objects.replace("{", "")
    objects = objects.replace("}", "")
    objects = objects.replace("[", "")
    objects = objects.replace("]", "")
    objects = objects.replace(" ", "")
    objects = objects.replace("bbox=", "")
    objects = objects.replace("bbox_size=", "")
    objects = objects.split(",")

    res_dict = {}
    for obj in objects:
        obj_feat = obj.split("=")
        k = obj_feat[0]
        v = obj_feat[1]
        try:
            res_dict[k].append(v)
        except:
            res_dict[k] = [v]

    return res_dict


def img_size_processing(row):
    img_size = row.img_size
    img_size = img_size.replace("{", "")
    img_size = img_size.replace("}", "")
    img_size = img_size.replace(" ", "")
    img_size = img_size.split(",")

    res_dict = {}
    for img_feat in img_size:
        img_feat_ = img_feat.split("=")
        k = img_feat_[0]
        v = img_feat_[1]
        res_dict[k] = v

    return res_dict
    

def df_processing(df):
    df = df[df.object_cnt != 0]
    df = df.reset_index(drop=True)

    df["objects.class"] = df.apply(lambda row: objects_processing(row)["class"], axis=1)
    df["objects.bbox.x_min"] = df.apply(lambda row: objects_processing(row)["x_min"], axis=1)
    df["objects.bbox.y_min"] = df.apply(lambda row: objects_processing(row)["y_min"], axis=1)
    df["objects.bbox.x_max"] = df.apply(lambda row: objects_processing(row)["x_max"], axis=1)
    df["objects.bbox.y_max"] = df.apply(lambda row: objects_processing(row)["y_max"], axis=1)
    df["objects.bbox_size.width"] = df.apply(lambda row: objects_processing(row)["width"], axis=1)

    df["img_size.width"] = df.apply(lambda row: img_size_processing(row)["width"], axis=1)
    df["img_size.height"] = df.apply(lambda row: img_size_processing(row)["height"], axis=1)

    return df


def get_boto3_client(service_name="s3", region_name="ap-northeast-2", aws_access_key_id=None, aws_secret_access_key=None):
    return boto3.client(    
        service_name=service_name, 
        region_name=region_name,
        aws_access_key_id=aws_access_key_id, 
        aws_secret_access_key=aws_secret_access_key
    )


def download_files_from_s3_bucket(boto3_client, bucket_name, list_files_path, target_dir=""):
    # download 할 때 split?
    for key in tqdm(list_files_path.keys()):
        file_len = len(list_files_path[key])
        for index, file_path in enumerate(list_files_path[key]):
            file_name = file_path.split("/")[-1]

            # store file path를 train, val 로 저장되도록 수정
            if index < file_len * 0.8:
                store_file_path = f"{target_dir}/train/{file_name}"
            else:
                store_file_path = f"{target_dir}/val/{file_name}"

            try:
                boto3_client.download_file(bucket_name, file_path, store_file_path)
            except:
                print(f"'{file_path}' download failed!!")
            
    print(f"finish download_files_from_s3_bucket")

def get_csv_path(dataset_id):
    
    response = requests.get(
    url=f"https://wksh34vatnkusizxf4gqf64p4q0onwbw.lambda-url.ap-northeast-2.on.aws/v0.1/api/dataset/train/get/id?dataset_id={dataset_id}",
    headers={
        'accept': 'application/json',
        'x-token': 'nota-its',
    })
    print(response.json())
    return response.json()['csv_path']
        
def get_data_from_s3(
    dataset_id,
    aws_access_key_id=None, 
    aws_secret_access_key=None,
    bucket_name="after-labeled",
    data_dir="/data"
):
    csv_s3_path = get_csv_path(dataset_id)
    csv_file_path = "./data.csv"
    img_dir = f"{data_dir}"
    
    os.makedirs(img_dir, exist_ok=True)
    
    s3 = get_boto3_client(aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key)

    print(csv_s3_path)

    s3.download_file("nota-its-athena-query-results", csv_s3_path, csv_file_path)
    
    data_df = pd.read_csv(csv_file_path, converters={'partition_4': str})
    data_df = df_processing(data_df)
    path_prefix = 'data/image'
    
    img_list = {}
    for idx, row in data_df.iterrows():
        cam = row["partition_4"]

        if (isinstance(cam, int)) is True:
            if cam < 10:
                path = f'{path_prefix}/{row["partition_1"]}/{row["partition_2"]}/{row["partition_3"]}/0{cam}/{row["partition_5"]}'
        else:
            if any(st.isalpha() for st in cam):
                path = f'{path_prefix}/{row["partition_1"]}/{row["partition_2"]}/{row["partition_3"]}/{cam}/{row["partition_5"]}'
            else:
                if int(cam) < 10:
                    cam_num = int(cam)
                    path = f'{path_prefix}/{row["partition_1"]}/{row["partition_2"]}/{row["partition_3"]}/0{cam_num}/{row["partition_5"]}'

        # key에 맞게 추가하기
        key = row['site']
        if key in img_list:
            img_list[key].append(f'{path}/{row["img_name"]}')
        else:
            img_list[key] = [f'{path}/{row["img_name"]}']
        
    download_files_from_s3_bucket(
        boto3_client=s3, 
        bucket_name=bucket_name, 
        list_files_path=img_list, 
        target_dir=img_dir
    )


if __name__ == "__main__":
    input_dataset_id= os.environ["DATASET_ID"]
    aws_access_key_id = os.environ["AWS_ACCESS_KEY_ID"]
    aws_secret_access_key = os.environ["AWS_SECRET_ACCESS_KEY"]
    
    get_data_from_s3(
        dataset_id=input_dataset_id,
        aws_access_key_id=aws_access_key_id, 
        aws_secret_access_key=aws_secret_access_key,
        bucket_name="after-labeled",
        data_dir="./dataset_name/imgs"
    )
