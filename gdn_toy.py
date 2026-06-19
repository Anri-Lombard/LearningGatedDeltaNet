import torch

torch.set_printoptions(precision=3, sci_mode=False)


def gdn_step(S, k, v, q, alpha, beta):
    """
    S:     [d_value, d_key]
    k:     [d_key]
    v:     [d_value]
    q:     [d_key]
    alpha: scalar tensor or float
    beta:  scalar tensor or float
    """
    S_decayed = alpha * S

    read_k = S_decayed @ k          # what memory currently returns for k
    error = v - read_k              # delta/error
    delta_S = beta * torch.outer(error, k)

    S_new = S_decayed + delta_S
    y = S_new @ q                   # read output using q

    return y, S_new, {
        "S_decayed": S_decayed,
        "read_k": read_k,
        "error": error,
        "delta_S": delta_S,
    }


S0 = torch.tensor([
    [5.0, 0.0],
    [0.0, 10.0],
])

k = torch.tensor([1.0, 0.0])
v = torch.tensor([7.0, 0.0])

alpha = 0.8
beta = 0.5

q = torch.tensor([1.0, 1.0])

y, S1, trace = gdn_step(S0, k, v, q, alpha, beta)

print("S_decayed:\n", trace["S_decayed"])
print("read_k:", trace["read_k"])
print("error:", trace["error"])
print("delta_S:\n", trace["delta_S"])
print("S1:\n", S1)
print("y:", y)
