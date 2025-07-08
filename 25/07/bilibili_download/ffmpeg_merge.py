import argparse
import glob
import subprocess
import os


def run_ffmpeg(input_file1, input_file2, output_file) -> bool:
    try:
        cmd = [
            "ffmpeg",
            "-hwaccel",
            "cuda",
            "-i",
            input_file1,
            "-i",
            input_file2,
            "-c:a",
            "copy",
            "-c:s",
            "copy",
            "-c:v",
            "h264_nvenc",
            "-preset",
            "p7",
            "-tune",
            "hq",
            output_file,
        ]

        # 执行命令
        print("正在执行FFmpeg命令...")
        # result = subprocess.run(
        #     cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        # )
        result = subprocess.run(cmd, check=True)

        print(f"FFmpeg命令执行成功！输出文件：{output_file}")
        print("标准输出：")
        print(result.stdout)

    except subprocess.CalledProcessError as e:
        print(f"错误：FFmpeg命令执行失败，返回代码 {e.returncode}")
        print("错误输出：")
        print(e.stderr)
        return False
    except FileNotFoundError:
        print("错误：未找到FFmpeg。请确保FFmpeg已安装并添加到系统PATH中。")
        return False

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ffmpeg merge tool",
        epilog="示例: python ffmpeg_merge.py source_folder",
    )
    parser.add_argument("source_folder", help="文件路径")
    input_folder = parser.parse_args().source_folder
    output_folder = os.path.join(input_folder, "output")
    os.makedirs(output_folder, exist_ok=True)
    for file_name in glob.glob(os.path.join(input_folder, "*_2.m4s")):
        print(file_name)
        file_name2 = file_name.replace("_2.m4s", ".m4s")
        base_name = os.path.basename(file_name)
        merge_ok = run_ffmpeg(
            file_name,
            file_name2,
            os.path.join(output_folder, f"{base_name.split('-')[0]}.mp4"),
        )
        # 只有运行成功后才会删除文件
        if merge_ok:
            os.remove(file_name)
            os.remove(file_name2)
