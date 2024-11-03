# author: Yunqi Li
# contact: liyunqixa@gmail.com
import subprocess
import pathlib


class DSU:
    '''
    Find-Union Set
    '''

    def __init__(self, size):
        self.fa = list(range(size))

    def find(self, x):
        if self.fa[x] != x:
            self.fa[x] = self.find(self.fa[x])
        return self.fa[x]

    def union(self, x, y):
        self.fa[self.find(x)] = self.find(y)
        
        
def run_cosine_exe(vec_path, vec_label_path, tau):
    cur_parent_dir = str(pathlib.Path(__file__).parent.resolve())
    cosine_exe_path = '/'.join([cur_parent_dir, "cosine", "cosine"])
    cmd_args = [cosine_exe_path, vec_path, vec_label_path, str(tau)]
    try:
        cosine_res = subprocess.run(cmd_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                                    check=True)
    except subprocess.CalledProcessError as cosine_error:
        print(cosine_error.returncode)
        print(cosine_error.output)
        raise
    
    print(f"cosine exists with: {cosine_res.returncode}")