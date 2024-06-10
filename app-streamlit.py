import os
import streamlit as st

def main():

    st.write("# Welcome to Streamlit! 👋")

    #AT:ee3ecc6b6cc257755dfe60af7f82dc98b9fed765
    # download internlm2 to the base_path directory using git tool
    base_path = './Werewolf-Kill-With-AI'
    os.system(f'git clone https://Junco:ee3ecc6b6cc257755dfe60af7f82dc98b9fed765@code.openxlab.org.cn/Junco/Werewolf-Kill-With-AI.git {base_path}')
    os.system(f'git lfs install')
    os.system(f'cd {base_path} && git lfs pull')

    #/home/xlab-app-center
    os.system('streamlit run ./web_demo.py --server.address=0.0.0.0 --server.port 7860')


if __name__ == "__main__":
    main()