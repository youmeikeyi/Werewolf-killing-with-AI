import os
import streamlit as st
from datetime import datetime
import pandas as pd
def main():

    st.write("# Welcome to Streamlit! 👋")

    #AT:ee3ecc6b6cc257755dfe60af7f82dc98b9fed765
    # download internlm2 to the base_path directory using git tool
    base_path = './internlm2-chat-7b'
    # base_path = './Werewolf-Kill-With-AI'
    # download repo to the base_path directory using git
    os.system(f'apt install git')
    os.system(f'apt install git-lfs')
    # os.system(f'git clone https://Junco:ee3ecc6b6cc257755dfe60af7f82dc98b9fed765@code.openxlab.org.cn/Junco/Werewolf-Kill-With-AI.git {base_path}')
    os.system(f'git clone https://code.openxlab.org.cn/Junco/Werewolf-Kill-With-AI.git {base_path}')

    os.system(f'cd {base_path} && git lfs pull')
    os.system(f'lmdeploy lite auto_awq \
        {base_path}\
      --calib-dataset ptb \
      --calib-samples 128 \
      --calib-seqlen 1024 \
      --w-bits 4 \
      --w-group-size 128 \
      --work-dir /root/internlm2-chat-1_8b-4bit')

    #/home/xlab-app-center
    # os.system('streamlit run ./web_demo.py --server.address=0.0.0.0 --server.port 7860')

if __name__ == "__main__":
    main()