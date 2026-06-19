## Core idea

The state update is a gated rank-1 correction:

```math
\tilde{S}_t
= \alpha_t S_{t-1}
+ \beta_t \left(v_t - \alpha_t S_{t-1} k_t\right) k_t^\top
```

## One memory example

Let:

```math
S_0 =
\begin{bmatrix}
0 & 0 \\
0 & 0
\end{bmatrix},
\quad
k =
\begin{bmatrix}
1 \\
0
\end{bmatrix},
\quad
v =
\begin{bmatrix}
2 \\
0
\end{bmatrix},
\quad
\alpha = 1,
\quad
\beta = 1
```

Now:

```math
k k^\top
=
\begin{bmatrix}
1 \\
0
\end{bmatrix}
\begin{bmatrix}
1 & 0
\end{bmatrix}
=
\begin{bmatrix}
1 & 0 \\
0 & 0
\end{bmatrix}
```

So:

```math
I - k k^\top
=
\begin{bmatrix}
1 & 0 \\
0 & 1
\end{bmatrix}
-
\begin{bmatrix}
1 & 0 \\
0 & 0
\end{bmatrix}
=
\begin{bmatrix}
0 & 0 \\
0 & 1
\end{bmatrix}
```

This is the erase part. It erases the old content associated with key $k$, while leaving the other slot alone.

The new content is written with:

```math
v k^\top
=
\begin{bmatrix}
2 \\
0
\end{bmatrix}
\begin{bmatrix}
1 & 0
\end{bmatrix}
=
\begin{bmatrix}
2 & 0 \\
0 & 0
\end{bmatrix}
```

Think of $v k^\top$ as making one column per entry in $k^\top$:

```math
v k^\top
=
\begin{bmatrix}
v \cdot 1 & v \cdot 0
\end{bmatrix}
```

So the first column gets one copy of $v$:

```math
v \cdot 1
=
\begin{bmatrix}
2 \\
0
\end{bmatrix}
```

And the second column gets zero copies of $v$:

```math
v \cdot 0
=
\begin{bmatrix}
0 \\
0
\end{bmatrix}
```

Therefore:

```math
v k^\top
=
\begin{bmatrix}
2 & 0 \\
0 & 0
\end{bmatrix}
```

This is the write part. It writes $v$ into the slot selected by $k$.

Now combine the erase and write:

```math
S_1 = S_0(I - k k^\top) + v k^\top
```

Substitute the matrices:

```math
S_1
=
\begin{bmatrix}
0 & 0 \\
0 & 0
\end{bmatrix}
\begin{bmatrix}
0 & 0 \\
0 & 1
\end{bmatrix}
+
\begin{bmatrix}
2 & 0 \\
0 & 0
\end{bmatrix}
```

Since $S_0$ is all zeros:

```math
S_1
=
\begin{bmatrix}
0 & 0 \\
0 & 0
\end{bmatrix}
+
\begin{bmatrix}
2 & 0 \\
0 & 0
\end{bmatrix}
=
\begin{bmatrix}
2 & 0 \\
0 & 0
\end{bmatrix}
```

Now read with the same key:

```math
S_1 k
=
\begin{bmatrix}
2 & 0 \\
0 & 0
\end{bmatrix}
\begin{bmatrix}
1 \\
0
\end{bmatrix}
=
\begin{bmatrix}
2 \\
0
\end{bmatrix}
```

So reading with $k$ returns the value that was written:

```math
S_1 k = v
```

## What this shows

This example shows the basic memory operation:

1. $I - k k^\top$ erases the old value at the slot selected by $k$.
2. $v k^\top$ writes the new value $v$ into that same slot.
3. Multiplying by $k$ reads the value back out.

So the state matrix $S$ acts like a tiny key-value memory. The key $k$ chooses where to write or read, and the value $v$ is the content stored there.

## Example 2: update the same key

Suppose later the model sees the same key:

```math
k =
\begin{bmatrix}
1 \\
0
\end{bmatrix}
```

But now it wants to store a new value:

```math
v =
\begin{bmatrix}
5 \\
0
\end{bmatrix}
```

With:

```math
\alpha = 1,
\quad
\beta = 1
```

We get:

```math
S_2 = S_1(I - k k^\top) + v k^\top
```

Substitute the variables:

```math
S_2
=
\begin{bmatrix}
2 & 0 \\
0 & 0
\end{bmatrix}
\begin{bmatrix}
0 & 0 \\
0 & 1
\end{bmatrix}
+
\begin{bmatrix}
5 \\
0
\end{bmatrix}
\begin{bmatrix}
1 & 0
\end{bmatrix}
```

The erase part gives:

```math
\begin{bmatrix}
2 & 0 \\
0 & 0
\end{bmatrix}
\begin{bmatrix}
0 & 0 \\
0 & 1
\end{bmatrix}
=
\begin{bmatrix}
0 & 0 \\
0 & 0
\end{bmatrix}
```

The write part gives:

```math
\begin{bmatrix}
5 \\
0
\end{bmatrix}
\begin{bmatrix}
1 & 0
\end{bmatrix}
=
\begin{bmatrix}
5 & 0 \\
0 & 0
\end{bmatrix}
```

So:

```math
S_2
=
\begin{bmatrix}
0 & 0 \\
0 & 0
\end{bmatrix}
+
\begin{bmatrix}
5 & 0 \\
0 & 0
\end{bmatrix}
=
\begin{bmatrix}
5 & 0 \\
0 & 0
\end{bmatrix}
```

## Why this matters

The important part is that the old value is replaced:

```math
2 \rightarrow 5
```

It does not add the new value on top of the old value:

```math
2 + 5 = 7
```

Plain additive linear attention would do:

```math
S_2^{\text{add}}
= S_1 + v k^\top
=
\begin{bmatrix}
2 & 0 \\
0 & 0
\end{bmatrix}
+
\begin{bmatrix}
5 & 0 \\
0 & 0
\end{bmatrix}
=
\begin{bmatrix}
7 & 0 \\
0 & 0
\end{bmatrix}
```

So reading with the same key would return:

```math
S_2^{\text{add}} k
=
\begin{bmatrix}
7 \\
0
\end{bmatrix}
```

That is accumulation, not replacement.

The delta rule instead writes the difference between the desired value and the value currently read from memory:

```math
S_t
= S_{t-1} + \left(v_t - S_{t-1}k_t\right)k_t^\top
```

Here:

```math
S_1 k =
\begin{bmatrix}
2 \\
0
\end{bmatrix}
```

But the new desired value is:

```math
v =
\begin{bmatrix}
5 \\
0
\end{bmatrix}
```

So the correction is:

```math
v - S_1 k
=
\begin{bmatrix}
5 \\
0
\end{bmatrix}
-
\begin{bmatrix}
2 \\
0
\end{bmatrix}
=
\begin{bmatrix}
3 \\
0
\end{bmatrix}
```

Equivalently, the current prediction error is:

```math
S_1 k - v
=
\begin{bmatrix}
2 \\
0
\end{bmatrix}
-
\begin{bmatrix}
5 \\
0
\end{bmatrix}
=
\begin{bmatrix}
-3 \\
0
\end{bmatrix}
```

The update adjusts $S$ in the opposite direction of that error:

```math
v - S_1 k = -(S_1 k - v)
```

Then the model adds only the correction:

```math
\left(v - S_1 k\right)k^\top
=
\begin{bmatrix}
3 \\
0
\end{bmatrix}
\begin{bmatrix}
1 & 0
\end{bmatrix}
=
\begin{bmatrix}
3 & 0 \\
0 & 0
\end{bmatrix}
```

So:

```math
S_2
=
\begin{bmatrix}
2 & 0 \\
0 & 0
\end{bmatrix}
+
\begin{bmatrix}
3 & 0 \\
0 & 0
\end{bmatrix}
=
\begin{bmatrix}
5 & 0 \\
0 & 0
\end{bmatrix}
```

That is the delta rule: do not blindly add the new value. First read what is already there, compare the current prediction $S_{t-1}k_t$ to the target $v_t$, then adjust $S$ based on the prediction error.

## Example 3: adding the gate

Now suppose the memory already has two slots:

```math
S =
\begin{bmatrix}
5 & 0 \\
0 & 10
\end{bmatrix}
```

The first key retrieves the first value:

```math
S
\begin{bmatrix}
1 \\
0
\end{bmatrix}
=
\begin{bmatrix}
5 \\
0
\end{bmatrix}
```

The second key retrieves the second value:

```math
S
\begin{bmatrix}
0 \\
1
\end{bmatrix}
=
\begin{bmatrix}
0 \\
10
\end{bmatrix}
```

Now update the first key:

```math
k =
\begin{bmatrix}
1 \\
0
\end{bmatrix},
\quad
v =
\begin{bmatrix}
7 \\
0
\end{bmatrix},
\quad
\alpha = 0.5,
\quad
\beta = 1
```

The gated delta update is:

```math
S_{\text{new}}
= \alpha S + \beta \left(v - \alpha S k\right) k^\top
```

Since $\beta = 1$, this can be rearranged:

```math
S_{\text{new}}
= \alpha S - \alpha S k k^\top + v k^\top
```

Factor out $S$:

```math
S_{\text{new}}
= S\left(\alpha(I - k k^\top)\right) + v k^\top
```

With $\alpha = 0.5$:

```math
S_{\text{new}}
= S\left(0.5(I - k k^\top)\right) + v k^\top
```

We already know:

```math
I - k k^\top
=
\begin{bmatrix}
0 & 0 \\
0 & 1
\end{bmatrix}
```

So:

```math
0.5(I - k k^\top)
=
\begin{bmatrix}
0 & 0 \\
0 & 0.5
\end{bmatrix}
```

The erase-and-decay part is:

```math
S\left(0.5(I - k k^\top)\right)
=
\begin{bmatrix}
5 & 0 \\
0 & 10
\end{bmatrix}
\begin{bmatrix}
0 & 0 \\
0 & 0.5
\end{bmatrix}
=
\begin{bmatrix}
0 & 0 \\
0 & 5
\end{bmatrix}
```

This clears the selected slot and decays the unselected slot.

The write part is:

```math
v k^\top
=
\begin{bmatrix}
7 \\
0
\end{bmatrix}
\begin{bmatrix}
1 & 0
\end{bmatrix}
=
\begin{bmatrix}
7 & 0 \\
0 & 0
\end{bmatrix}
```

So:

```math
S_{\text{new}}
=
\begin{bmatrix}
0 & 0 \\
0 & 5
\end{bmatrix}
+
\begin{bmatrix}
7 & 0 \\
0 & 0
\end{bmatrix}
=
\begin{bmatrix}
7 & 0 \\
0 & 5
\end{bmatrix}
```

The gate $\alpha = 0.5$ decays what is not selected by $k$. The selected slot is erased and rewritten to $7$, while the unselected slot decays from $10$ to $5$.

## What the gated delta update gives us

This example shows two things happening at once:

1. The gate/decay part forgets old state:

```math
\alpha S
```

2. The delta part edits a specific key-value association:

```math
\left(v - \alpha S k\right)k^\top
```

So Gated DeltaNet is not just decaying memory. It can decay the overall state and also perform a targeted edit at the key $k$.

A Mamba-style decay-only update is closer to:

```math
S_{\text{new}} = \alpha S + \text{new input}
```

That can forget globally by shrinking old state, but it does not explicitly say:

```math
\text{replace the value currently stored at this key}
```

Gated DeltaNet does have that edit operation. It reads the current value at key $k$, compares it to the desired value $v$, and writes the correction back into that same key:

```math
v - \alpha S k
```

So the useful picture is:

- $\alpha$ controls global forgetting.
- $k$ selects which association to edit.
- $v$ is the new value we want at that key.
- The delta term changes only what is needed to make the readout closer to $v$.

That is why it behaves like an editable associative memory, not just a decaying hidden state.
