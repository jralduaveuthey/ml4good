# see instructions in original file: C:\Users\jraldua-veuthey\Documents\Github\ml4good\mlab\w0d1_instructions.md

########################################## Tips ##########################################

# %%
import math
from einops import rearrange, repeat, reduce
import torch as t

# %%

def assert_all_equal(actual: t.Tensor, expected: t.Tensor) -> None:
    assert actual.shape == expected.shape, f"Shape mismatch, got: {actual.shape}"
    assert (actual == expected).all(), f"Value mismatch, got: {actual}"
    print("Passed!")


def assert_all_close(actual: t.Tensor, expected: t.Tensor, rtol=1e-05, atol=0.0001) -> None:
    assert actual.shape == expected.shape, f"Shape mismatch, got: {actual.shape}"
    assert t.allclose(actual, expected, rtol=rtol, atol=atol)
    print("Passed!")


def rearrange_1() -> t.Tensor:
    """Return the following tensor using only torch.arange and einops.rearrange:

    [[3, 4],
    [5, 6],
    [7, 8]]
    """
    v = t.arange(3, 9)
    v2 = rearrange(v, "(a b) -> a b", a=3, b=2)
    return v2
    # pass


expected = t.tensor([[3, 4], [5, 6], [7, 8]])
assert_all_equal(rearrange_1(), expected)

# %%

def rearrange_2() -> t.Tensor:
    """Return the following tensor using only torch.arange and einops.rearrange:

    [[1, 2, 3],
    [4, 5, 6]]
    """
    v = t.arange(1,7)
    v2 = rearrange(v, "(a b) -> a b", a=2, b=3)
    return v2
    # pass


assert_all_equal(rearrange_2(), t.tensor([[1, 2, 3], [4, 5, 6]]))

# %%

def rearrange_3() -> t.Tensor:
    """Return the following tensor using only torch.arange and einops.rearrange:

    [[[1], [2], [3], [4], [5], [6]]]
    """
    v = t.arange(1,7)
    v2 = rearrange(v, "(a b c) -> a b c", a=1, b=6, c=1)
    return v2
    # pass


assert_all_equal(rearrange_3(), t.tensor([[[1], [2], [3], [4], [5], [6]]]))
# %%

########################################## Creating Tensors: Tensor vs tensor ##########################################

# %%

x = t.arange(5)
y = t.Tensor(x.shape)
y2 = t.Tensor(tuple(x.shape))
y3 = t.Tensor(list(x.shape))
print(y, y2, y3)

# What should this cell print?
x = t.Tensor([False, True])
print(x.dtype)

try:
    print(t.tensor([1, 2, 3, 4]).mean())
except Exception as e:
    print("Exception raised: ", e)
# %%
########################################## Other good ways to create tensors are: ##########################################

def temperatures_average(temps: t.Tensor) -> t.Tensor:
    """Return th each week.

    temps: a 1D temperature containing temperatures for each day.
    Length will be a multiple of 7 and the first 7 days are for the first week, second 7 days for the second week, etc.

    You can do this with a single call to reduce.
    """
    assert len(temps) % 7 == 0

    # # alternative 1
    # n_weeks = len(temps)//7
    # temps1 = rearrange(temps, "(a b) -> a b", a=n_weeks, b=7)
    # tempsM = temps1.mean(dim=1)
    # return tempsM

    # alternative 2
    return reduce(temps, "(h 7) -> h", "mean")

# Pattern: The pattern (h 7) -> h instructs einops.reduce on how to transform the input tensor. This pattern tells einops to:
# Group the elements of temps into chunks of 7 (h 7), where each chunk represents one week. The h here is a placeholder representing the number of weeks (or groups) and 7 is the number of days in each week.
# The -> h part specifies the desired output shape, indicating that for each group of 7 days, a single value (the mean temperature for that week) should be produced, resulting in a tensor with a length equal to the number of weeks.
# Reduction Operation: "mean" specifies the reduction operation to be applied to each group of 7 days. It calculates the average (mean) temperature for each week.

    # pass


temps = t.tensor([71, 72, 70, 75, 71, 72, 70, 68, 65, 60, 68, 60, 55, 59, 75, 80, 85, 80, 78, 72, 83], dtype=t.float32)
expected = t.tensor([71.5714, 62.1429, 79.0])
assert_all_close(temperatures_average(temps), expected)

# %%
def assert_all_close(actual: t.Tensor, expected: t.Tensor, rtol=1e-05, atol=0.0001) -> None:
    assert actual.shape == expected.shape, f"Shape mismatch, got: {actual.shape}"
    assert t.allclose(actual, expected, rtol=rtol, atol=atol)
    print("Passed!")

def temperatures_differences(temps: t.Tensor) -> t.Tensor:
    """For each day, subtract the average for the week the day belongs to.

    temps: as above
    """
    assert len(temps) % 7 == 0

    # # alternative 1
    # tempsM = temperatures_average(temps)
    # n_weeks = len(temps)//7
    # temps1 = rearrange(temps, "(a b) -> a b", a=7, b=n_weeks)
    # temps2 = temps1 - tempsM
    # temps3 = temps2.flatten()
    # return temps3

    # alternative 2
    avg = repeat(temperatures_average(temps), "w -> w 7") #this returns a tensor with dimensions (3, 7)
    avg = repeat(temperatures_average(temps), "w -> (w 7)") #this returns a tensor with dimensions (1, 21)
    return temps - avg

temps = t.tensor([71, 72, 70, 75, 71, 72, 70, 68, 65, 60, 68, 60, 55, 59, 75, 80, 85, 80, 78, 72, 83], dtype=t.float32)
expected = t.tensor(
    [
        -0.5714,
        0.4286,
        -1.5714,
        3.4286,
        -0.5714,
        0.4286,
        -1.5714,
        5.8571,
        2.8571,
        -2.1429,
        5.8571,
        -2.1429,
        -7.1429,
        -3.1429,
        -4.0,
        1.0,
        6.0,
        1.0,
        -1.0,
        -7.0,
        4.0,
    ]
)
actual = temperatures_differences(temps)
assert_all_close(actual, expected)

# %%

def temperatures_normalized(temps: t.Tensor) -> t.Tensor:
    """For each day, subtract the weekly average and divide by the weekly standard deviation.

    temps: as above

    Pass torch.std to reduce.
    """
    # pass

    # alternative 1
    number_weeks = len(temps)//7
    tempsn_wx7 = rearrange(temps, "(n_w a) -> n_w a", n_w = number_weeks)
    avg1xn_w = tempsn_wx7.mean(dim=1)
    # avg1xn_w = reduce(temps, "(n_w 7) -> n_w", "mean")
    avg7xn_w = repeat(avg1xn_w, "n_w -> 7 n_w")
    temps7xn_w = tempsn_wx7.T 
    diff7xn_w =  temps7xn_w- avg7xn_w
    std1xn_w = reduce(temps, "(n_w 7) -> n_w", t.std)
    std7xn_w = repeat(std1xn_w, "n_w -> 7 n_w")
    final7xn_w = diff7xn_w / std7xn_w
    final = final7xn_w.T.flatten()

    # alternative 2
    avg = repeat(temperatures_average(temps), "w -> (w 7)") #this is the new thing in alternative 2
    number_weeks = len(temps)//7
    tempsn_wx7 = rearrange(temps, "(n_w a) -> n_w a", n_w = number_weeks)
    temps7xn_w = tempsn_wx7.T 
    avgn_wx7 = rearrange(avg, "(n_w a) -> n_w a", n_w = number_weeks)
    avg7xn_w = avgn_wx7.T
    temps7xn_w = tempsn_wx7.T 
    diff7xn_w =  temps7xn_w- avg7xn_w
    std1xn_w = reduce(temps, "(n_w 7) -> n_w", t.std)
    std7xn_w = repeat(std1xn_w, "n_w -> 7 n_w")
    final7xn_w = diff7xn_w / std7xn_w
    final = final7xn_w.T.flatten()

    # alternative 3
    avg = repeat(temperatures_average(temps), "w -> (w 7)")
    std = repeat(reduce(temps, "(h 7) -> h", t.std), "w -> (w 7)") #this is the new thing in alternative 3
    final = (temps - avg) / std

    return final

temps = t.tensor([71, 72, 70, 75, 71, 72, 70, 68, 65, 60, 68, 60, 55, 59, 75, 80, 85, 80, 78, 72, 83], dtype=t.float32)
expected = t.tensor(
    [
        -0.3326,
        0.2494,
        -0.9146,
        1.9954,
        -0.3326,
        0.2494,
        -0.9146,
        1.1839,
        0.5775,
        -0.4331,
        1.1839,
        -0.4331,
        -1.4438,
        -0.6353,
        -0.8944,
        0.2236,
        1.3416,
        0.2236,
        -0.2236,
        -1.5652,
        0.8944,
    ]
)
actual = temperatures_normalized(temps)
assert_all_close(actual, expected)

# %%

def batched_dot_product_nd(a: t.Tensor, b: t.Tensor) -> t.Tensor:
    """Return the batched dot product of a and b, where the first dimension is the batch dimension.

    That is, out[i] = dot(a[i], b[i]) for i in 0..len(a).
    a and b can have any number of dimensions greater than 1.

    a: shape (b, i_1, i_2, ..., i_n)
    b: shape (b, i_1, i_2, ..., i_n)

    Returns: shape (b, )

    Use torch.einsum. You can use the ellipsis "..." in the einsum formula to represent an arbitrary number of dimensions.
    """
    assert a.shape == b.shape
    # pass
    return t.einsum('b... , b... -> b', a, b) 




actual = batched_dot_product_nd(t.tensor([[1, 1, 0], [0, 0, 1]]), t.tensor([[1, 1, 0], [1, 1, 0]]))
expected = t.tensor([2, 0])
assert_all_equal(actual, expected)
actual2 = batched_dot_product_nd(t.arange(12).reshape((3, 2, 2)), t.arange(12).reshape((3, 2, 2)))
expected2 = t.tensor([14, 126, 366])
assert_all_equal(actual2, expected2)

# %%

def identity_matrix(n: int) -> t.Tensor:
    """Return the identity matrix of size nxn.

    Don't use torch.eye or similar.

    Hint: you can do it with arange, rearrange, and ==.
    Bonus: find a different way to do it.
    """
    assert n >= 0
    # pass
    
    # alternative 1
    n2 = n*n
    v = t.arange(1, n2+1)
    M = rearrange(v,"(a b) -> a b", a=n, b=n)

    out = M==M.T

    # alternative 2
    out = (rearrange(t.arange(n), "i->i 1") == t.arange(n)).float()

    return out


assert_all_equal(identity_matrix(3), t.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]]))
assert_all_equal(identity_matrix(0), t.zeros((0, 0)))

# %%

def sample_distribution(probs: t.Tensor, n: int) -> t.Tensor:
    """Return n random samples from probs, where probs is a normalized probability distribution.

    probs: shape (k,) where probs[i] is the probability of event i occurring.
    n: number of random samples

    Return: shape (n,) where out[i] is an integer indicating which event was sampled.

    Use torch.rand and torch.cumsum to do this without any explicit loops.

    Note: if you think your solution is correct but the test is failing, try increasing the value of n.
    """
    assert abs(probs.sum() - 1.0) < 0.001
    assert (probs >= 0).all()
    # pass
    # lp = len(probs)
    # v = t.rand(n)*lp #this returns a tensor with n random numbers between 0 and lp
    # out = v.int() #this returns a tensor with n random ints between 0 and lp
    # return out

    # alternative gpt4 - https://chat.openai.com/share/d099541e-3eb6-4b2c-a678-659ee0894270
    # Generate n random numbers uniformly distributed between 0 and 1
    random_samples = t.rand(n)
    # Compute the cumulative sum of probs to get the CDF
    cdf = t.cumsum(probs, dim=0)
    # For each random sample, find the index of the first element in the CDF that is greater than or equal to it
    out = t.searchsorted(cdf, random_samples)

    # alternative mlab (I do not understand it) - https://chat.openai.com/share/d099541e-3eb6-4b2c-a678-659ee0894270
    out =  (t.rand(n, 1) > t.cumsum(probs, dim=0)).sum(dim=-1)
    
    return out


n = 10000000
probs = t.tensor([0.05, 0.1, 0.1, 0.2, 0.15, 0.4])
freqs = t.bincount(sample_distribution(probs, n)) / n
assert_all_close(freqs, probs, rtol=0.001, atol=0.001)

# %%

def classifier_accuracy(scores: t.Tensor, true_classes: t.Tensor) -> t.Tensor:
    """Return the fraction of inputs for which the maximum score corresponds to the true class for that input.

    scores: shape (batch, n_classes). A higher score[b, i] means that the classifier thinks class i is more likely.
    true_classes: shape (batch, ). true_classes[b] is an integer from [0...n_classes).

    Use torch.argmax.
    """
    assert true_classes.max() < scores.shape[1]
    # pass
    pred_classes = t.argmax(scores, dim=1)
    n_classes = len(true_classes)
    out = (pred_classes == true_classes).sum() / n_classes

    # alternative mlab
    out = (scores.argmax(dim=1) == true_classes).float().mean()
    return out

scores = t.tensor([[0.75, 0.5, 0.25], [0.1, 0.5, 0.4], [0.1, 0.7, 0.2]])
true_classes = t.tensor([0, 1, 0])
expected = 2.0 / 3.0
assert classifier_accuracy(scores, true_classes) == expected

# %%

def total_price_indexing(prices: t.Tensor, items: t.Tensor) -> float:
    """Given prices for each kind of item and a tensor of items purchased, return the total price.

    prices: shape (k, ). prices[i] is the price of the ith item.
    items: shape (n, ). A 1D tensor where each value is an item index from [0..k).

    Use integer array indexing. The below document describes this for NumPy but it's the same in PyTorch:

    https://numpy.org/doc/stable/user/basics.indexing.html#integer-array-indexing
    """
    assert items.max() < prices.shape[0]
    # pass
    out = 0
    for i in range(len(items)):
        out += prices[items[i]]

    # alternative mlab
    out = prices[items].sum().item()
    return out
        


prices = t.tensor([0.5, 1, 1.5, 2, 2.5])
items = t.tensor([0, 0, 1, 1, 4, 3, 2])
assert total_price_indexing(prices, items) == 9.0

# %%

def gather_2d(matrix: t.Tensor, indexes: t.Tensor) -> t.Tensor:
    """Perform a gather operation along the second dimension.

    matrix: shape (m, n)
    indexes: shape (m, k)

    Return: shape (m, k). out[i][j] = matrix[i][indexes[i][j]]

    For this problem, the test already passes and it's your job to write at least three asserts relating the arguments and the output. This is a tricky function and worth spending some time to wrap your head around its behavior.

    See: https://pytorch.org/docs/stable/generated/torch.gather.html?highlight=gather#torch.gather
    """
    # https://machinelearningknowledge.ai/how-to-use-torch-gather-function-in-pytorch-with-examples/
    "TODO: YOUR CODE HERE"
    assert matrix.ndim == indexes.ndim
    assert indexes.shape[0] <= matrix.shape[0]
    out = matrix.gather(1, indexes)
    "TODO: YOUR CODE HERE"
    assert out.shape == indexes.shape
    return out


matrix = t.arange(15).view(3, 5)
indexes = t.tensor([[4], [3], [2]])
expected = t.tensor([[4], [8], [12]])
assert_all_equal(gather_2d(matrix, indexes), expected)
indexes2 = t.tensor([[2, 4], [1, 3], [0, 2]])
expected2 = t.tensor([[2, 4], [6, 8], [10, 12]])
assert_all_equal(gather_2d(matrix, indexes2), expected2)

# %%

def total_price_gather(prices: t.Tensor, items: t.Tensor) -> float:
    """Compute the same as total_price_indexing, but use torch.gather."""
    assert items.max() < prices.shape[0]
    # pass
    paid_prices = t.gather(input=prices, dim=0, index=items)
    out = paid_prices.sum().item()

    # alternative mlab
    out = prices.gather(0, items).sum().item()
    return out


prices = t.tensor([0.5, 1, 1.5, 2, 2.5])
items = t.tensor([0, 0, 1, 1, 4, 3, 2])
assert total_price_gather(prices, items) == 9.0

# %%

def integer_array_indexing(matrix: t.Tensor, coords: t.Tensor) -> t.Tensor:
    """Return the values at each coordinate using integer array indexing.

    For details on integer array indexing, see:
    https://numpy.org/doc/stable/user/basics.indexing.html#integer-array-indexing

    matrix: shape (d_0, d_1, ..., d_n)
    coords: shape (batch, n)

    Return: (batch, )
    """
    # pass
    # alternative mlab
    out = matrix[tuple(coords.T)] # I do not fully understand this but here some explanation: https://chat.openai.com/share/e40afd7d-f889-4f13-a35f-5a60a3be68e8
    return out


mat_2d = t.arange(15).view(3, 5)
coords_2d = t.tensor([[0, 1], [0, 4], [1, 4]])
actual = integer_array_indexing(mat_2d, coords_2d)
assert_all_equal(actual, t.tensor([1, 4, 9]))
mat_3d = t.arange(2 * 3 * 4).view((2, 3, 4))
coords_3d = t.tensor([[0, 0, 0], [0, 1, 1], [0, 2, 2], [1, 0, 3], [1, 2, 0]])
actual = integer_array_indexing(mat_3d, coords_3d)
assert_all_equal(actual, t.tensor([0, 5, 10, 15, 20]))

# %%

def batched_logsumexp(matrix: t.Tensor) -> t.Tensor:
    """For each row of the matrix, compute log(sum(exp(row))) in a numerically stable way.

    matrix: shape (batch, n)

    Return: (batch, ). For each i, out[i] = log(sum(exp(matrix[i]))).

    Do this without using PyTorch's logsumexp function.

    A couple useful blogs about this function:
    - https://leimao.github.io/blog/LogSumExp/ -> useful 
    - https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/
    """
    # pass
    max_values, _ = matrix.max(dim=1) 
    a = max_values
    # Reshape max_values for broadcasting by adding an extra dimension
    a_reshaped = rearrange(a, "n -> n 1")
    out = a + (matrix - a_reshaped).exp().sum(dim=1).log()

    # alternative mlab
    C = matrix.max(dim=-1).values
    exps = t.exp(matrix - rearrange(C, "n -> n 1"))
    out = C + t.log(t.sum(exps, dim=-1))

    return out


matrix = t.tensor([[-1000, -1000, -1000, -1000], [1000, 1000, 1000, 1000]])
expected = t.tensor([-1000 + math.log(4), 1000 + math.log(4)])
actual = batched_logsumexp(matrix)
print("expected", expected)
print("actual", actual)
assert_all_close(actual, expected)
matrix2 = t.randn((10, 20))
expected2 = t.logsumexp(matrix2, dim=-1)
actual2 = batched_logsumexp(matrix2)
assert_all_close(actual2, expected2)

# %%

def batched_softmax(matrix: t.Tensor) -> t.Tensor:
    """For each row of the matrix, compute softmax(row).

    Do this without using PyTorch's softmax function.
    Instead, use the definition of softmax: https://en.wikipedia.org/wiki/Softmax_function

    matrix: shape (batch, n)

    Return: (batch, n). For each i, out[i] should sum to 1.
    """
    # pass
    exps = matrix.exp()
    exp_sums = exps.sum(dim=1)
    exp_sums_col = rearrange(exp_sums, "n -> n 1")
    out = exps / exp_sums_col 

    # alternative mlab
    exp = matrix.exp()
    out = exp / exp.sum(dim=-1, keepdim=True)

    return out 


matrix = t.arange(1, 6).view((1, 5)).float().log()
expected = t.arange(1, 6).view((1, 5)) / 15.0
actual = batched_softmax(matrix)
assert_all_close(actual, expected)
for i in [0.12, 3.4, -5, 6.7]:
    assert_all_close(actual, batched_softmax(matrix + i))
matrix2 = t.rand((10, 20))
actual2 = batched_softmax(matrix2)
assert actual2.min() >= 0.0
assert actual2.max() <= 1.0
assert_all_equal(actual2.argsort(), matrix2.argsort())
assert_all_close(actual2.sum(dim=-1), t.ones(matrix2.shape[:-1]))

# %%

def batched_logsoftmax(matrix: t.Tensor) -> t.Tensor:
    """Compute log(softmax(row)) for each row of the matrix.

    matrix: shape (batch, n)

    Return: (batch, n). For each i, exp(out[i]) should sum to 1.

    Do this without using PyTorch's logsoftmax function.
    For each row, subtract the maximum first to avoid overflow if the row contains large values.
    """
    # pass
    out = matrix - batched_logsumexp(matrix)

    return out

matrix = t.arange(1, 6).view((1, 5)).float()
start = 1000
matrix2 = t.arange(start + 1, start + 6).view((1, 5)).float()
actual = batched_logsoftmax(matrix2)
expected = t.tensor([[-4.4519, -3.4519, -2.4519, -1.4519, -0.4519]])
assert_all_close(actual, expected)

# %%

def batched_cross_entropy_loss(logits: t.Tensor, true_labels: t.Tensor) -> t.Tensor:
    """Compute the cross entropy loss for each example in the batch.

    logits: shape (batch, classes). logits[i][j] is the unnormalized prediction for example i and class j.
    true_labels: shape (batch, ). true_labels[i] is an integer index representing the true class for example i.

    Return: shape (batch, ). out[i] is the loss for example i.

    Hint: convert the logits to log-probabilities using your batched_logsoftmax from above.
    Then the loss for an example is just the negative of the log-probability that the model assigned to the true class. Use torch.gather to perform the indexing.
    """
    # pass
    assert logits.shape[0] == true_labels.shape[0]
    assert true_labels.max() < logits.shape[1]

    log_probs = batched_logsoftmax(logits)
    out2D = -t.gather(log_probs, 1, true_labels.unsqueeze(1))
    out = rearrange(out2D, "n 1 -> n")

    # alternative mlab
    logprobs = batched_logsoftmax(logits)
    indices = rearrange(true_labels, "n -> n 1")
    pred_at_index = logprobs.gather(1, indices)
    out =  -rearrange(pred_at_index, "n 1 -> n")
    return out
    #here the solution from mlab is the same but there is an AssertionError in the test...dunno why


logits = t.tensor([[float("-inf"), float("-inf"), 0], [1 / 3, 1 / 3, 1 / 3], [float("-inf"), 0, 0]])
true_labels = t.tensor([2, 0, 0])
expected = t.tensor([0.0, math.log(3), float("inf")])
actual = batched_cross_entropy_loss(logits, true_labels)
assert_all_close(actual, expected)

# %%

def collect_rows(matrix: t.Tensor, row_indexes: t.Tensor) -> t.Tensor:
    """Return a 2D matrix whose rows are taken from the input matrix in order according to row_indexes.

    matrix: shape (m, n)
    row_indexes: shape (k,). Each value is an integer in [0..m).

    Return: shape (k, n). out[i] is matrix[row_indexes[i]].
    """
    assert row_indexes.max() < matrix.shape[0]
    # pass
    return matrix[row_indexes]


matrix = t.arange(15).view((5, 3))
row_indexes = t.tensor([0, 2, 1, 0])
actual = collect_rows(matrix, row_indexes)
expected = t.tensor([[0, 1, 2], [6, 7, 8], [3, 4, 5], [0, 1, 2]])
assert_all_equal(actual, expected)

# %%

def collect_columns(matrix: t.Tensor, column_indexes: t.Tensor) -> t.Tensor:
    """Return a 2D matrix whose columns are taken from the input matrix in order according to column_indexes.

    matrix: shape (m, n)
    column_indexes: shape (k,). Each value is an integer in [0..n).

    Return: shape (m, k). out[:, i] is matrix[:, column_indexes[i]].
    """
    assert column_indexes.max() < matrix.shape[1]
    # pass
    matrixT = matrix.T
    temp = matrixT[column_indexes]
    out = temp.T

    # alternative mlab
    out = matrix[:, column_indexes]
    return out  


matrix = t.arange(15).view((5, 3))
column_indexes = t.tensor([0, 2, 1, 0])
actual = collect_columns(matrix, column_indexes)
expected = t.tensor([[0, 2, 1, 0], [3, 5, 4, 3], [6, 8, 7, 6], [9, 11, 10, 9], [12, 14, 13, 12]])
assert_all_equal(actual, expected)

# %%

########################################## Practice with torch.as_strided ##########################################
# NOTE: The resources links provided by the tutorial are very confusing since in the end in this exercise I don't have to think in terms of bytes or bits. I have to first set the size argument to the same size as the output tensor and then I have to calculate the stride. There's gonna be as many strides as dimensions in the size and each entry for a stride is how many elements it moves across a dimension. The examples in 2D output tensors are very helpful to understand this. I do not understand in higher dimensions

# See https://chat.openai.com/share/2686c52d-fe04-40c1-b540-20029e6138bf

from collections import namedtuple

TestCase = namedtuple("TestCase", ["output", "size", "stride"])
test_input_a = t.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13, 14], [15, 16, 17, 18, 19]])
test_cases = [
    TestCase(output=t.tensor([0, 1, 2, 3]), size=(4,), stride=(1,)), #This creates a 1-dimensional tensor of size 4. The stride of 1 means it will take consecutive elements from the original tensor without skipping any, starting from the first element. Thus, it takes the first four elements of the first row.

    TestCase(output=t.tensor([[0, 1, 2], [5, 6, 7]]), size=(2,3), stride=(5,1)), #This specifies a 2x3 tensor. The stride (5, 1) means for the row dimension, it jumps 5 elements to start the next row, which essentially moves down a row in the original tensor (since it has 5 columns). The column stride of 1 takes consecutive elements within the row. Thus, it forms two rows: the first row of the original tensor and the second row of the original tensor, each with the first three elements.

    TestCase(output=t.tensor([[0, 0, 0], [11, 11, 11]]), size=(2,3), stride=(11,0)), #"jump 11 elements to start the next row" and "do not move when changing columns,"

    TestCase(output=t.tensor([0, 6, 12, 18]), size=(4,), stride=(6,)),  #"jump 6 elements to start the next row" 

    TestCase(output=t.tensor([[[0, 1, 2]], [[9, 10, 11]]]), size=(2,1,3), stride=(9,0,1)),# This case specifies a tensor of shape 2x1x3. The stride (9, 0, 1) means it jumps 9 elements to get to the next 2D slice, does not move to get to the next row within each 2D slice, and moves 1 element to get to the next column within each row.

    TestCase(
        output=t.tensor([[[[0, 1], [2, 3]], [[4, 5], [6, 7]]], [[[12, 13], [14, 15]], [[16, 17], [18, 19]]]]),
        size=(2, 2, 2, 2),
        stride=(12, 4, 2, 1),
        # Move 12 elements to go to the next 3D block.
        # Move 4 elements to go to the next 2D slice within each 3D block.
        # Move 2 elements to go to the next row within each 2D slice.
        # Move 1 element to go to the next column within each row.
    ),
]

for (i, case) in enumerate(test_cases):
    actual = test_input_a.as_strided(size=case.size, stride=case.stride)
    if (case.output != actual).any():
        print(f"Test {i} failed:")
        print(f"Expected: {case.output}")
        print(f"**********Actual: {actual}")
        print("-------------------")	
    else:
        print(f"Test {i} passed!")
        print(f"Expected: {case.output}")
        print(f">>>>>>>>>Actual: {actual}")
        print("-------------------")	


a = t.tensor([[1, 2, 3], [4, 5, 6]])
view_with_stride_custom = t.as_strided(a, size=(1, 3), stride=(2, 1))
print(view_with_stride_custom)

# %%

########################################## Implementing ReLU ##########################################
        
def test_relu(relu_func):
    print(f"Testing: {relu_func.__name__}")
    x = t.arange(-1, 3, dtype=t.float32, requires_grad=True)
    out = relu_func(x)
    expected = t.tensor([0.0, 0.0, 1.0, 2.0])
    assert_all_close(out, expected)


def relu_clone_setitem(x: t.Tensor) -> t.Tensor:
    """Make a copy with torch.clone and then assign to parts of the copy."""
    # pass
    xclone = x.clone()
    xclone[xclone < 0] = 0
    return xclone


test_relu(relu_clone_setitem)

# %%

def relu_where(x: t.Tensor) -> t.Tensor:
    """Use torch.where."""
    # pass
    return t.where(x>0, x, t.tensor(0.0))


test_relu(relu_where)

# %%

def relu_maximum(x: t.Tensor) -> t.Tensor:
    """Use torch.maximum."""
    # pass
    return t.maximum(x,t.tensor(0.0))


test_relu(relu_maximum)

# %%

def relu_abs(x: t.Tensor) -> t.Tensor:
    """Use torch.abs."""
    # pass

    # return t.abs(x) * (x >= 0).float()

    # alternative mlab
    return (x.abs() + x) / 2.0


test_relu(relu_abs)

# %%

def relu_multiply_bool(x: t.Tensor) -> t.Tensor:
    """Create a boolean tensor and multiply the input by it elementwise."""
    # pass
    return (x >= 0) * x


test_relu(relu_multiply_bool)
