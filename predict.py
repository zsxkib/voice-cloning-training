# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import time
import zipfile
from cog import BasePredictor, Input, Path as CogPath
from concurrent.futures import ThreadPoolExecutor
from random import shuffle
import json
import os
import pathlib
from subprocess import Popen, PIPE, STDOUT
import numpy as np
import faiss
import os
import subprocess
import shutil
import glob
from zipfile import ZipFile

# List of URLs and destinations
downloads = [
    (
        "https://weights.replicate.delivery/default/rvc/assets/pretrained/D32k.pth",
        "assets/pretrained/D32k.pth",
    ),
    (
        "https://weights.replicate.delivery/default/rvc/assets/pretrained/D40k.pth",
        "assets/pretrained/D40k.pth",
    ),
    (
        "https://weights.replicate.delivery/default/rvc/assets/pretrained/D48k.pth",
        "assets/pretrained/D48k.pth",
    ),
    (
        "https://weights.replicate.delivery/default/rvc/assets/pretrained/G32k.pth",
        "assets/pretrained/G32k.pth",
    ),
    (
        "https://weights.replicate.delivery/default/rvc/assets/pretrained/G40k.pth",
        "assets/pretrained/G40k.pth",
    ),
    (
        "https://weights.replicate.delivery/default/rvc/assets/pretrained/G48k.pth",
        "assets/pretrained/G48k.pth",
    ),
    (
        "https://weights.replicate.delivery/default/rvc/assets/pretrained/f0D32k.pth",
        "assets/pretrained/f0D32k.pth",
    ),
    (
        "https://weights.replicate.delivery/default/rvc/assets/pretrained/f0D40k.pth",
        "assets/pretrained/f0D40k.pth",
    ),
    (
        "https://weights.replicate.delivery/default/rvc/assets/pretrained/f0D48k.pth",
        "assets/pretrained/f0D48k.pth",
    ),
    (
        "https://weights.replicate.delivery/default/rvc/assets/pretrained/f0G32k.pth",
        "assets/pretrained/f0G32k.pth",
    ),
    (
        "https://weights.replicate.delivery/default/rvc/assets/pretrained/f0G40k.pth",
        "assets/pretrained/f0G40k.pth",
    ),
    (
        "https://weights.replicate.delivery/default/rvc/assets/pretrained/f0G48k.pth",
        "assets/pretrained/f0G48k.pth",
    ),
    (
        "https://weights.replicate.delivery/default/rvc/assets/pretrained_v2/D32k.pth",
        "assets/pretrained_v2/D32k.pth",
    ),
    (
        "https://weights.replicate.delivery/default/rvc/assets/pretrained_v2/D40k.pth",
        "assets/pretrained_v2/D40k.pth",
    ),
    (
        "https://weights.replicate.delivery/default/rvc/assets/pretrained_v2/D48k.pth",
        "assets/pretrained_v2/D48k.pth",
    ),
    (
        "https://weights.replicate.delivery/default/rvc/assets/pretrained_v2/G32k.pth",
        "assets/pretrained_v2/G32k.pth",
    ),
    (
        "https://weights.replicate.delivery/default/rvc/assets/pretrained_v2/G40k.pth",
        "assets/pretrained_v2/G40k.pth",
    ),
    (
        "https://weights.replicate.delivery/default/rvc/assets/pretrained_v2/G48k.pth",
        "assets/pretrained_v2/G48k.pth",
    ),
    (
        "https://weights.replicate.delivery/default/rvc/assets/pretrained_v2/f0D32k.pth",
        "assets/pretrained_v2/f0D32k.pth",
    ),
    (
        "https://weights.replicate.delivery/default/rvc/assets/pretrained_v2/f0D40k.pth",
        "assets/pretrained_v2/f0D40k.pth",
    ),
    (
        "https://weights.replicate.delivery/default/rvc/assets/pretrained_v2/f0D48k.pth",
        "assets/pretrained_v2/f0D48k.pth",
    ),
    (
        "https://weights.replicate.delivery/default/rvc/assets/pretrained_v2/f0G32k.pth",
        "assets/pretrained_v2/f0G32k.pth",
    ),
    (
        "https://weights.replicate.delivery/default/rvc/assets/pretrained_v2/f0G40k.pth",
        "assets/pretrained_v2/f0G40k.pth",
    ),
    (
        "https://weights.replicate.delivery/default/rvc/assets/pretrained_v2/f0G48k.pth",
        "assets/pretrained_v2/f0G48k.pth",
    ),
    (
        "https://weights.replicate.delivery/default/rvc/assets/hubert/hubert_base.pt",
        "assets/hubert/hubert_base.pt",
    ),
    (
        "https://weights.replicate.delivery/default/rvc/assets/rmvpe/rmvpe.pt",
        "assets/rmvpe/rmvpe.pt",
    ),
]


def infer_folder_name(base_path):
    # Print the current working directory and base path
    print(f"Current working directory: {os.getcwd()}")
    print(f"Base path: {base_path}")

    # Check if the directory exists
    if not os.path.isdir(base_path):
        print(f"Directory does not exist: {base_path}")
        return None

    # List all directories in the base_path
    dirs = [
        d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))
    ]

    # Return the first directory name
    return dirs[0] if dirs else None


def execute_command(command):
    process = subprocess.Popen(command, shell=True)
    output, error = process.communicate()

    if process.returncode != 0:
        print(f"Error occurred: {error}")
    else:
        print(f"Output: {output}")

    return output, error


def train_index(exp_dir1, version19):
    exp_dir = "logs/%s" % (exp_dir1)
    os.makedirs(exp_dir, exist_ok=True)
    feature_dir = (
        "%s/3_feature256" % (exp_dir)
        if version19 == "v1"
        else "%s/3_feature768" % (exp_dir)
    )
    if not os.path.exists(feature_dir):
        # return "请先进行特征提取!"
        return "Please perform feature extraction first!"

    listdir_res = list(os.listdir(feature_dir))
    if len(listdir_res) == 0:
        # return "请先进行特征提取！"
        return "Please perform feature extraction first!"
    infos = []
    npys = []
    for name in sorted(listdir_res):
        phone = np.load("%s/%s" % (feature_dir, name))
        npys.append(phone)
    big_npy = np.concatenate(npys, 0)
    big_npy_idx = np.arange(big_npy.shape[0])
    np.random.shuffle(big_npy_idx)
    big_npy = big_npy[big_npy_idx]
    if big_npy.shape[0] > 2e5:
        infos.append("Trying doing kmeans %s shape to 10k centers." % big_npy.shape[0])
        yield "\n".join(infos)
        try:
            big_npy = (
                MiniBatchKMeans(
                    n_clusters=10000,
                    verbose=True,
                    batch_size=256 * config.n_cpu,
                    compute_labels=False,
                    init="random",
                )
                .fit(big_npy)
                .cluster_centers_
            )
        except:
            info = traceback.format_exc()
            logger.info(info)
            infos.append(info)
            yield "\n".join(infos)

    np.save("%s/total_fea.npy" % exp_dir, big_npy)
    n_ivf = min(int(16 * np.sqrt(big_npy.shape[0])), big_npy.shape[0] // 39)
    infos.append("%s,%s" % (big_npy.shape, n_ivf))
    yield "\n".join(infos)
    index = faiss.index_factory(256 if version19 == "v1" else 768, "IVF%s,Flat" % n_ivf)
    # index = faiss.index_factory(256if version19=="v1"else 768, "IVF%s,PQ128x4fs,RFlat"%n_ivf)
    infos.append("training")
    yield "\n".join(infos)
    index_ivf = faiss.extract_index_ivf(index)  #
    index_ivf.nprobe = 1
    index.train(big_npy)
    faiss.write_index(
        index,
        "%s/trained_IVF%s_Flat_nprobe_%s_%s_%s.index"
        % (exp_dir, n_ivf, index_ivf.nprobe, exp_dir1, version19),
    )

    infos.append("adding")
    yield "\n".join(infos)
    batch_size_add = 8192
    for i in range(0, big_npy.shape[0], batch_size_add):
        index.add(big_npy[i : i + batch_size_add])
    faiss.write_index(
        index,
        "%s/added_IVF%s_Flat_nprobe_%s_%s_%s.index"
        % (exp_dir, n_ivf, index_ivf.nprobe, exp_dir1, version19),
    )
    infos.append(
        "Successfully built index, added_IVF%s_Flat_nprobe_%s_%s_%s.index"
        % (n_ivf, index_ivf.nprobe, exp_dir1, version19)
    )


def click_train(
    exp_dir1,
    sr2,
    if_f0_3,
    spk_id5,
    save_epoch10,
    total_epoch11,
    batch_size12,
    if_save_latest13,
    pretrained_G14,
    pretrained_D15,
    gpus16,
    if_cache_gpu17,
    if_save_every_weights18,
    version19,
):
    # 生成filelist
    exp_dir = "%s/logs/%s" % (".", exp_dir1)
    os.makedirs(exp_dir, exist_ok=True)
    gt_wavs_dir = "%s/0_gt_wavs" % (exp_dir)
    feature_dir = (
        "%s/3_feature256" % (exp_dir)
        if version19 == "v1"
        else "%s/3_feature768" % (exp_dir)
    )
    if if_f0_3:
        f0_dir = "%s/2a_f0" % (exp_dir)
        f0nsf_dir = "%s/2b-f0nsf" % (exp_dir)
        names = (
            set([name.split(".")[0] for name in os.listdir(gt_wavs_dir)])
            & set([name.split(".")[0] for name in os.listdir(feature_dir)])
            & set([name.split(".")[0] for name in os.listdir(f0_dir)])
            & set([name.split(".")[0] for name in os.listdir(f0nsf_dir)])
        )
    else:
        names = set([name.split(".")[0] for name in os.listdir(gt_wavs_dir)]) & set(
            [name.split(".")[0] for name in os.listdir(feature_dir)]
        )
    opt = []
    for name in names:
        if if_f0_3:
            opt.append(
                "%s/%s.wav|%s/%s.npy|%s/%s.wav.npy|%s/%s.wav.npy|%s"
                % (
                    gt_wavs_dir.replace("\\", "\\\\"),
                    name,
                    feature_dir.replace("\\", "\\\\"),
                    name,
                    f0_dir.replace("\\", "\\\\"),
                    name,
                    f0nsf_dir.replace("\\", "\\\\"),
                    name,
                    spk_id5,
                )
            )
        else:
            opt.append(
                "%s/%s.wav|%s/%s.npy|%s"
                % (
                    gt_wavs_dir.replace("\\", "\\\\"),
                    name,
                    feature_dir.replace("\\", "\\\\"),
                    name,
                    spk_id5,
                )
            )
    fea_dim = 256 if version19 == "v1" else 768
    if if_f0_3:
        for _ in range(2):
            opt.append(
                "%s/logs/mute/0_gt_wavs/mute%s.wav|%s/logs/mute/3_feature%s/mute.npy|%s/logs/mute/2a_f0/mute.wav.npy|%s/logs/mute/2b-f0nsf/mute.wav.npy|%s"
                % (".", sr2, ".", fea_dim, ".", ".", spk_id5)
            )
    else:
        for _ in range(2):
            opt.append(
                "%s/logs/mute/0_gt_wavs/mute%s.wav|%s/logs/mute/3_feature%s/mute.npy|%s"
                % (".", sr2, ".", fea_dim, spk_id5)
            )
    shuffle(opt)
    with open("%s/filelist.txt" % exp_dir, "w") as f:
        f.write("\n".join(opt))

    # Replace logger.debug, logger.info with print statements
    print("Write filelist done")
    print("Use gpus:", str(gpus16))
    if pretrained_G14 == "":
        print("No pretrained Generator")
    if pretrained_D15 == "":
        print("No pretrained Discriminator")
    if version19 == "v1" or sr2 == "40k":
        config_path = "configs/v1/%s.json" % sr2
    else:
        config_path = "configs/v2/%s.json" % sr2
    config_save_path = os.path.join(exp_dir, "config.json")
    if not pathlib.Path(config_save_path).exists():
        with open(config_save_path, "w", encoding="utf-8") as f:
            with open(config_path, "r") as config_file:
                config_data = json.load(config_file)
                json.dump(
                    config_data,
                    f,
                    ensure_ascii=False,
                    indent=4,
                    sort_keys=True,
                )
            f.write("\n")

    cmd = (
        'python infer/modules/train/train.py -e "%s" -sr %s -f0 %s -bs %s -g %s -te %s -se %s %s %s -l %s -c %s -sw %s -v %s'
        % (
            exp_dir1,
            sr2,
            1 if if_f0_3 else 0,
            batch_size12,
            gpus16,
            total_epoch11,
            save_epoch10,
            "-pg %s" % pretrained_G14 if pretrained_G14 != "" else "",
            "-pd %s" % pretrained_D15 if pretrained_D15 != "" else "",
            1 if if_save_latest13 == True else 0,
            1 if if_cache_gpu17 == True else 0,
            1 if if_save_every_weights18 == True else 0,
            version19,
        )
    )
    # Use PIPE to capture the output and error streams
    p = Popen(
        cmd,
        shell=True,
        cwd=".",
        stdout=PIPE,
        stderr=STDOUT,
        bufsize=1,
        universal_newlines=True,
    )

    # Print the command's output as it runs
    for line in p.stdout:
        print(line.strip())

    # Wait for the process to finish
    p.wait()
    return "Training completed. You can check the training log in the console or the 'train.log' file in the experiment directory."


def download_weights(url, dest):
    # Check if the destination directory exists, if not, create it
    dest_dir = os.path.dirname(dest)
    os.makedirs(dest_dir, exist_ok=True)

    # Check if the destination file already exists
    if not os.path.exists(dest):
        start = time.time()
        print("Downloading URL: ", url)
        print("Downloading to: ", dest)
        subprocess.check_call(["pget", url, dest], close_fds=False)
        print("Downloading took: ", time.time() - start)
    else:
        print(f"File already exists: {dest}")


class Predictor(BasePredictor):
    def setup(self) -> None:
        # Running the downloads in parallel
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(download_weights, url, dest) for url, dest in downloads
            ]

        # Ensure all downloads are complete before proceeding
        for future in futures:
            future.result()

    def delete_old_files(self):
        os.makedirs("assets/weights", exist_ok=True)

        # Delete 'dataset' folder if it exists
        if os.path.exists("dataset"):
            shutil.rmtree("dataset")

        # Delete 'Model' folder if it exists
        if os.path.exists("Model"):
            shutil.rmtree("Model")

        # Delete contents of 'assets/weights' folder but keep the folder
        # if os.path.exists("assets/weights"):
        #     for filename in os.listdir("assets/weights"):
        #         file_path = os.path.join("assets/weights", filename)
        #         if os.path.isfile(file_path) or os.path.islink(file_path):
        #             os.unlink(file_path)
        #         elif os.path.isdir(file_path):
        #             shutil.rmtree(file_path)

        # Delete contents of 'logs' folder but keep the folder and 'mute' directory
        if os.path.exists("logs"):
            for filename in os.listdir("logs"):
                file_path = os.path.join("logs", filename)
                if filename == "mute":
                    continue  # Skip the 'mute' directory
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)

    def predict(
        self,
        dataset_zip: CogPath = Input(
            description="Upload dataset zip, zip should contain `dataset/<rvc_name>/split_<i>.wav`"
        ),
        sample_rate: str = Input(
            description="Sample rate", default="48k", choices=["40k", "48k"]
        ),
        version: str = Input(description="Version", default="v2", choices=["v1", "v2"]),
        f0method: str = Input(
            description="F0 method, `rmvpe_gpu` recommended.",
            default="rmvpe_gpu",
            choices=["pm", "dio", "harvest", "rmvpe", "rmvpe_gpu"],
        ),
        epoch: int = Input(description="Epoch", default=10),
        batch_size: str = Input(description="Batch size", default="7"),
    ) -> CogPath:
        self.delete_old_files()

        dataset_path = str(dataset_zip)
        with zipfile.ZipFile(dataset_path, "r") as zip_ref:
            zip_ref.extractall(".")

        # Create Model Folder
        model_name = infer_folder_name("dataset")
        sample_rate = "40000" if sample_rate == "40k" else "48000"
        dataset = "dataset/" + model_name
        exp_dir = model_name
        ksample_rate = "48k"
        ksample_rate = "40k" if sample_rate == "40000" else "48k"
        save_frequency = 50
        cache_gpu = True

        os.makedirs("%s/logs/%s" % (".", exp_dir), exist_ok=True)
        f = open("%s/logs/%s/preprocess.log" % (".", exp_dir), "w")
        os.makedirs("%s/logs/%s" % (".", exp_dir), exist_ok=True)
        f = open("%s/logs/%s/extract_f0_feature.log" % (".", exp_dir), "w")
        f.close()

        # Process Data
        command = f"python infer/modules/train/preprocess.py '{dataset}' {sample_rate} 2 './logs/{exp_dir}' False 3.0"
        print(command)
        execute_command(command)

        # Feature Extraction

        if f0method != "rmvpe_gpu":
            command = f"python infer/modules/train/extract/extract_f0_print.py './logs/{exp_dir}' 2 '{f0method}'"
        else:
            command = f"python infer/modules/train/extract/extract_f0_rmvpe.py 1 0 0 './logs/{exp_dir}' True"
        print(command)
        execute_command(command)

        command = f"python infer/modules/train/extract_feature_print.py cuda:0 1 0 0 './logs/{exp_dir}' '{version}'"
        print(command)
        execute_command(command)

        # Train Feature Index
        result_generator = train_index(exp_dir, version)
        for result in result_generator:
            print(result)

        # Train Model
        pretrained_paths = {
            "v1": {
                "40k": ("assets/pretrained/f0G40k.pth", "assets/pretrained/f0D40k.pth"),
                "48k": ("assets/pretrained/f0G48k.pth", "assets/pretrained/f0D48k.pth"),
            },
            "v2": {
                "40k": (
                    "assets/pretrained_v2/f0G40k.pth",
                    "assets/pretrained_v2/f0D40k.pth",
                ),
                "48k": (
                    "assets/pretrained_v2/f0G48k.pth",
                    "assets/pretrained_v2/f0D48k.pth",
                ),
            },
        }
        G_path, D_path = pretrained_paths[version][ksample_rate]

        result_generator = click_train(
            exp_dir,
            ksample_rate,
            True,
            0,
            save_frequency,
            epoch,
            batch_size,
            True,
            G_path,
            D_path,
            0,
            cache_gpu,
            False,
            version,
        )
        print(result_generator)

        # Create directory
        print("Creating directory...")
        os.makedirs(f"./Model/{exp_dir}", exist_ok=True)

        # Copy files
        print("Copying files...")
        for file in glob.glob(f"logs/{exp_dir}/added_*.index"):
            print(f"Copying file: {file}")
            shutil.copy(file, f"./Model/{exp_dir}")

        for file in glob.glob(f"logs/{exp_dir}/total_*.npy"):
            print(f"Copying file: {file}")
            shutil.copy(file, f"./Model/{exp_dir}")

        print(f"Copying file: assets/weights/{exp_dir}.pth")
        shutil.copy(f"assets/weights/{exp_dir}.pth", f"./Model/{exp_dir}")

        # Define the base directory
        print("Defining the base directory...")
        base_dir = os.path.abspath(f"./Model/{exp_dir}")

        # Create a Zip file
        print("Creating a Zip file...")
        zip_file_path = os.path.join(base_dir, f"{exp_dir}.zip")
        with ZipFile(zip_file_path, "w") as zipf:
            # Add 'added_*.index' files
            print("Adding 'added_*.index' files to the Zip file...")
            for file in glob.glob(os.path.join(base_dir, "added_*.index")):
                if os.path.exists(file):
                    print(f"Adding file: {file}")
                    zipf.write(file, arcname=os.path.basename(file))
                else:
                    print(f"File not found: {file}")

            # Add 'total_*.npy' files
            print("Adding 'total_*.npy' files to the Zip file...")
            for file in glob.glob(os.path.join(base_dir, "total_*.npy")):
                if os.path.exists(file):
                    print(f"Adding file: {file}")
                    zipf.write(file, arcname=os.path.basename(file))
                else:
                    print(f"File not found: {file}")

            # Add specific file
            print("Adding specific file to the Zip file...")
            exp_file = os.path.join(base_dir, f"{exp_dir}.pth")
            if os.path.exists(exp_file):
                print(f"Adding file: {exp_file}")
                zipf.write(exp_file, arcname=os.path.basename(exp_file))
            else:
                print(f"File not found: {exp_file}")

        print(f"Zip file path: {zip_file_path}")
        return CogPath(zip_file_path)
