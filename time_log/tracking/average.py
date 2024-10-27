import glob
import os

# 경로에서 모든 .txt 파일 가져오기
txt_files = glob.glob("*.txt")

# 결과를 저장할 파일 열기
with open("average.txt", "w") as output_file:
    for txt_file in txt_files:
        # 파일을 열고 숫자 값 읽기
        with open(txt_file, "r") as f:
            times = [float(line.strip()) for line in f if line.strip()]

        # 파일에서 최소, 최대, 평균 값 계산
        if times:
            min_time = min(times)
            max_time = max(times)
            avg_time = sum(times) / len(times)

            # 파일 이름 (확장자 제거) 및 통계 값 작성
            output_file.write(f"{os.path.splitext(txt_file)[0]}\n")
            output_file.write(f"min : {min_time:.7f} sec\n")
            output_file.write(f"max: {max_time:.7f} sec\n")
            output_file.write(f"average: {avg_time:.7f} sec\n\n")
