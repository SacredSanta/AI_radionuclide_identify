def polynomial_hash(str):
    p = 31
    m = 1_000_000_007
    hash_value = 0
    for char in str:
        hash_value = (hash_value * p + ord (char)) % m
        print(hash_value)
    return hash_value

def solution(string_list, query_list):
    hash_list = [polynomial_hash(str) for str in string_list]
    
    result = []
    for query in query_list:
        query_hash = polynomial_hash(query)
        if query_hash in hash_list:
            result.append(True)
        else:
            result.append(False)
    return result

string_list = ["apple", "banana", "cherry"]
query_list = ["banana", "kiwi", "melon", "apple"]

print(solution(string_list, query_list))





#%%
import numpy as np
np.random.seed(42)
m, k, n = 5, 4, 3

V = np.random.rand(k, n)
X = np.random.rand(m, n)
X_hat = np.random.rand(m, n)
A = np.random.rand(m, k)

for i in range(m):
    for l in range(k):
        temp_numer = 0
        temp_deno = 0
        for j in range(n):
            temp_numer += (V[l,j]*X[i,j])/ (X_hat[i,j]+1e-5)
            temp_deno += V[l,j]
        A[i,l] = A[i,l] * (temp_numer/temp_deno+1e-5)

# %%
import numpy as np
np.random.seed(42)
# 임의의 예제 데이터 생성
m, k, n = 5, 4, 3
V = np.random.rand(k, n)
X = np.random.rand(m, n)
X_hat = np.random.rand(m, n)
A = np.random.rand(m, k)

# 벡터화된 계산
# temp_numer: (m, k) 크기의 배열
temp_numer = np.dot(X / (X_hat + 1e-5), V.T)
# X : nxd   V : kxd


# temp_deno: (k, ) 크기의 배열
temp_deno = np.sum(V, axis=1)

# (m, k) 크기의 배열로 temp_deno를 확장
temp_deno_expanded = temp_deno[np.newaxis, :]

# 벡터화된 A 계산
A *= (temp_numer / temp_deno_expanded + 1e-5)
# %%







#%%
def preorder(nodes, idx):
    if idx < len(nodes):
        print(idx, "Go in")
        ret = str(nodes[idx]) + " "
        ret += preorder(nodes, idx * 2 + 1)
        print("Pre1")
        ret += preorder(nodes, idx * 2 + 2)
        print("Pre2")
        return ret
    else:
        print(idx, "Go end")
        return ""
    
def inorder(nodes, idx):
    if idx < len(nodes):
        ret = inorder(nodes, idx * 2 + 1)
        ret += str(nodes[idx]) + " "
        ret += inorder(nodes, idx * 2 + 2)
        return ret
    else:
        return ""
    
def postorder(nodes, idx):
    if idx < len(nodes):
        ret = postorder(nodes, idx * 2 + 1)
        ret += postorder(nodes, idx * 2 + 2)
        ret += str(nodes[idx]) + " "
        return ret
    else:
        return ""
    
def solution(nodes):
    return [
        preorder(nodes, 0)[:-1],
        inorder(nodes, 0)[:-1],
        postorder(nodes, 0)[:-1],
    ]




nodes = [1,2,3,4,5,6,7]
print(solution(nodes))
# %%
