# Math intuition

## Core idea

The state update is a gated rank-1 correction:

```math
\tilde{S}_t
= \alpha_t S_{t-1}
+ \beta_t \left(v_t - \alpha_t S_{t-1} k_t\right) k_t^\top
```

In one sentence: Gated DeltaNet keeps a compressed memory, reads what the memory currently returns for a key, computes the error against the desired value, then uses gates to control forgetting and correction strength.

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

## Example 4: adding the update-strength knob

Now keep the same memory and same target value, but turn down $\beta$:

```math
S =
\begin{bmatrix}
5 & 0 \\
0 & 10
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
7 \\
0
\end{bmatrix},
\quad
\alpha = 1,
\quad
\beta = 0.5
```

The gated delta update is:

```math
S_{\text{new}}
= \alpha S + \beta \left(v - \alpha S k\right) k^\top
```

Since $\alpha = 1$:

```math
S_{\text{new}}
= S + 0.5\left(v - S k\right) k^\top
```

First read the current value at the selected key:

```math
S k
=
\begin{bmatrix}
5 & 0 \\
0 & 10
\end{bmatrix}
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

The desired value is:

```math
v =
\begin{bmatrix}
7 \\
0
\end{bmatrix}
```

So the full correction would be:

```math
v - S k
=
\begin{bmatrix}
7 \\
0
\end{bmatrix}
-
\begin{bmatrix}
5 \\
0
\end{bmatrix}
=
\begin{bmatrix}
2 \\
0
\end{bmatrix}
```

But $\beta = 0.5$, so we only apply half of that correction:

```math
0.5(v - S k)
=
0.5
\begin{bmatrix}
2 \\
0
\end{bmatrix}
=
\begin{bmatrix}
1 \\
0
\end{bmatrix}
```

Write that half-correction back to the selected key:

```math
0.5(v - S k)k^\top
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
S_{\text{new}}
=
\begin{bmatrix}
5 & 0 \\
0 & 10
\end{bmatrix}
+
\begin{bmatrix}
1 & 0 \\
0 & 0
\end{bmatrix}
=
\begin{bmatrix}
6 & 0 \\
0 & 10
\end{bmatrix}
```

Reading with the same key now gives:

```math
S_{\text{new}}k
=
\begin{bmatrix}
6 & 0 \\
0 & 10
\end{bmatrix}
\begin{bmatrix}
1 \\
0
\end{bmatrix}
=
\begin{bmatrix}
6 \\
0
\end{bmatrix}
```

So $\beta$ controls how strongly the model moves the selected memory toward the target:

```math
5 \rightarrow 6
```

instead of jumping all the way:

```math
5 \rightarrow 7
```

We can also see this in the erase/write form. Starting from:

```math
S_{\text{new}}
= S + \beta(v - S k)k^\top
```

Expand:

```math
S_{\text{new}}
= S - \beta S k k^\top + \beta v k^\top
```

Factor out $S$:

```math
S_{\text{new}}
= S(I - \beta k k^\top) + \beta v k^\top
```

With $\beta = 0.5$:

```math
I - 0.5 k k^\top
=
\begin{bmatrix}
1 & 0 \\
0 & 1
\end{bmatrix}
-
0.5
\begin{bmatrix}
1 & 0 \\
0 & 0
\end{bmatrix}
=
\begin{bmatrix}
0.5 & 0 \\
0 & 1
\end{bmatrix}
```

The selected slot is only half-erased:

```math
S(I - 0.5 k k^\top)
=
\begin{bmatrix}
5 & 0 \\
0 & 10
\end{bmatrix}
\begin{bmatrix}
0.5 & 0 \\
0 & 1
\end{bmatrix}
=
\begin{bmatrix}
2.5 & 0 \\
0 & 10
\end{bmatrix}
```

And the new value is only half-written:

```math
0.5 v k^\top
=
0.5
\begin{bmatrix}
7 & 0 \\
0 & 0
\end{bmatrix}
=
\begin{bmatrix}
3.5 & 0 \\
0 & 0
\end{bmatrix}
```

Together:

```math
S_{\text{new}}
=
\begin{bmatrix}
2.5 & 0 \\
0 & 10
\end{bmatrix}
+
\begin{bmatrix}
3.5 & 0 \\
0 & 0
\end{bmatrix}
=
\begin{bmatrix}
6 & 0 \\
0 & 10
\end{bmatrix}
```

So with $\alpha = 1$ and $\beta = 0.5$, nothing globally decays. The unselected memory stays at $10$. Only the selected association is softly edited, moving halfway from the current value $5$ toward the desired value $7$.

The scalar intuition is:

```math
\text{new value}
= \text{old value} + \beta(\text{target value} - \text{old value})
```

For $\beta = 0.5$:

```math
6 = 5 + 0.5(7 - 5)
```

Now suppose $\beta = 0.25$ instead:

```math
\text{new value}
= 5 + 0.25(7 - 5)
= 5 + 0.25(2)
= 5 + 0.5
= 5.5
```

So the matrix update becomes:

```math
S_{\text{new}}
= S + 0.25(v - S k)k^\top
```

We already know:

```math
v - S k
=
\begin{bmatrix}
2 \\
0
\end{bmatrix}
```

Apply only one quarter of that correction:

```math
0.25(v - S k)
=
0.25
\begin{bmatrix}
2 \\
0
\end{bmatrix}
=
\begin{bmatrix}
0.5 \\
0
\end{bmatrix}
```

Write it back to the selected key:

```math
0.25(v - S k)k^\top
=
\begin{bmatrix}
0.5 \\
0
\end{bmatrix}
\begin{bmatrix}
1 & 0
\end{bmatrix}
=
\begin{bmatrix}
0.5 & 0 \\
0 & 0
\end{bmatrix}
```

So:

```math
S_{\text{new}}
=
\begin{bmatrix}
5 & 0 \\
0 & 10
\end{bmatrix}
+
\begin{bmatrix}
0.5 & 0 \\
0 & 0
\end{bmatrix}
=
\begin{bmatrix}
5.5 & 0 \\
0 & 10
\end{bmatrix}
```

With $\beta = 0.25$, the selected memory moves one quarter of the way from $5$ to $7$:

```math
5 \rightarrow 5.5
```

If $\beta = 0$, the update applies none of the correction:

```math
\text{new value}
= 5 + 0(7 - 5)
= 5
```

So:

```math
S_{\text{new}}
= S + 0(v - S k)k^\top
= S
=
\begin{bmatrix}
5 & 0 \\
0 & 10
\end{bmatrix}
```

The selected slot does not move at all:

```math
5 \rightarrow 5
```

## Worked example: compute the output

Let:

```math
S_0 =
\begin{bmatrix}
5 & 0 \\
0 & 10
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
7 \\
0
\end{bmatrix},
\quad
\alpha = 1,
\quad
\beta = 0.5
```

First compute the current read:

```math
S_0 k
=
\begin{bmatrix}
5 & 0 \\
0 & 10
\end{bmatrix}
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

The target is:

```math
v =
\begin{bmatrix}
7 \\
0
\end{bmatrix}
```

So the error is:

```math
v - S_0 k
=
\begin{bmatrix}
7 \\
0
\end{bmatrix}
-
\begin{bmatrix}
5 \\
0
\end{bmatrix}
=
\begin{bmatrix}
2 \\
0
\end{bmatrix}
```

Apply $\beta = 0.5$:

```math
\beta(v - S_0 k)
=
0.5
\begin{bmatrix}
2 \\
0
\end{bmatrix}
=
\begin{bmatrix}
1 \\
0
\end{bmatrix}
```

Write that correction at key $k$:

```math
\beta(v - S_0 k)k^\top
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

So $k$ determines both where to check memory and where to apply the edit:

```math
S_0 k
```

checks what memory currently returns for that key, and:

```math
\beta(v - S_0 k)k^\top
```

writes the correction back into that same key direction.

This also shows why the scale of $k$ matters. If $k$ is too large, then:

```math
S k
```

can become too large, and the correction:

```math
(v - S k)k^\top
```

can become too large too. That is why controls like normalization and gating are needed, so the key direction behaves stably.

Since $\alpha = 1$, the updated memory is:

```math
S_1
= S_0 + \beta(v - S_0 k)k^\top
=
\begin{bmatrix}
5 & 0 \\
0 & 10
\end{bmatrix}
+
\begin{bmatrix}
1 & 0 \\
0 & 0
\end{bmatrix}
=
\begin{bmatrix}
6 & 0 \\
0 & 10
\end{bmatrix}
```

Now compute the output by reading with $k$:

```math
y = S_1 k
```

So:

```math
y
=
\begin{bmatrix}
6 & 0 \\
0 & 10
\end{bmatrix}
\begin{bmatrix}
1 \\
0
\end{bmatrix}
=
\begin{bmatrix}
6 \\
0
\end{bmatrix}
```

The output moved halfway from the old read to the target:

```math
\begin{bmatrix}
5 \\
0
\end{bmatrix}
\rightarrow
\begin{bmatrix}
6 \\
0
\end{bmatrix}
```

If we read with the other query:

```math
q =
\begin{bmatrix}
0 \\
1
\end{bmatrix}
```

then:

```math
S_1 q
=
\begin{bmatrix}
6 & 0 \\
0 & 10
\end{bmatrix}
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

So with $\alpha = 1$, the second memory slot is unchanged.

Now suppose:

```math
\alpha = 0.8,
\quad
\beta = 0.5
```

The update is:

```math
S_1
= \alpha S_0 + \beta(v - \alpha S_0 k)k^\top
```

First decay the memory:

```math
\alpha S_0
=
0.8
\begin{bmatrix}
5 & 0 \\
0 & 10
\end{bmatrix}
=
\begin{bmatrix}
4 & 0 \\
0 & 8
\end{bmatrix}
```

Now read the selected key after decay:

```math
\alpha S_0 k
=
\begin{bmatrix}
4 & 0 \\
0 & 8
\end{bmatrix}
\begin{bmatrix}
1 \\
0
\end{bmatrix}
=
\begin{bmatrix}
4 \\
0
\end{bmatrix}
```

The correction is:

```math
v - \alpha S_0 k
=
\begin{bmatrix}
7 \\
0
\end{bmatrix}
-
\begin{bmatrix}
4 \\
0
\end{bmatrix}
=
\begin{bmatrix}
3 \\
0
\end{bmatrix}
```

Apply $\beta = 0.5$:

```math
\beta(v - \alpha S_0 k)
=
0.5
\begin{bmatrix}
3 \\
0
\end{bmatrix}
=
\begin{bmatrix}
1.5 \\
0
\end{bmatrix}
```

Write the correction:

```math
\beta(v - \alpha S_0 k)k^\top
=
\begin{bmatrix}
1.5 \\
0
\end{bmatrix}
\begin{bmatrix}
1 & 0
\end{bmatrix}
=
\begin{bmatrix}
1.5 & 0 \\
0 & 0
\end{bmatrix}
```

So:

```math
S_1
=
\begin{bmatrix}
4 & 0 \\
0 & 8
\end{bmatrix}
+
\begin{bmatrix}
1.5 & 0 \\
0 & 0
\end{bmatrix}
=
\begin{bmatrix}
5.5 & 0 \\
0 & 8
\end{bmatrix}
```

Reading with $k$ gives:

```math
S_1 k
=
\begin{bmatrix}
5.5 & 0 \\
0 & 8
\end{bmatrix}
\begin{bmatrix}
1 \\
0
\end{bmatrix}
=
\begin{bmatrix}
5.5 \\
0
\end{bmatrix}
```

Reading with $q$ gives:

```math
S_1 q
=
\begin{bmatrix}
5.5 & 0 \\
0 & 8
\end{bmatrix}
\begin{bmatrix}
0 \\
1
\end{bmatrix}
=
\begin{bmatrix}
0 \\
8
\end{bmatrix}
```

So $\alpha = 0.8$ decays the unselected memory slot:

```math
\begin{bmatrix}
0 \\
10
\end{bmatrix}
\rightarrow
\begin{bmatrix}
0 \\
8
\end{bmatrix}
```

# Comparing Gated DeltaNet to a Transformer

A Transformer keeps separate key and value vectors for many previous tokens.

For token $i$, it stores something like:

```math
k_i, v_i
```

Then at a later token $t$, the current query $q_t$ compares itself against the previous keys:

```math
q_t^\top k_1,\quad q_t^\top k_2,\quad \ldots,\quad q_t^\top k_{t-1}
```

So the Transformer is always asking:

```text
Which previous token should I attend to?
```

It has explicit memory of the previous tokens. The upside is that it can point back to a specific old token very directly. The downside is that the key/value cache grows with sequence length.

Gated DeltaNet does something different. It keeps one compressed memory matrix:

```math
S_t
```

Past tokens are folded into this matrix through updates like:

```math
S_t
= \alpha_t S_{t-1}
+ \beta_t \left(v_t - \alpha_t S_{t-1} k_t\right) k_t^\top
```

So at token $t$, it is not keeping every old token as a separate key/value pair. Instead, the current key or query retrieves from the learned memory matrix:

```math
S_t q_t
```

The question becomes:

```text
What does my current key/query retrieve from the memory matrix?
```

So the tradeoff is:

- Transformer: explicit memory of past tokens.
- Gated DeltaNet: compressed editable memory.

The Transformer can retrieve an old token directly, but its cache grows with sequence length.

Gated DeltaNet is cheaper for long context because the memory size can stay fixed, but old information can interfere or get compressed together inside $S_t$.

## Transformer weighted read

More specifically, a Transformer output is a weighted sum of previous value vectors:

```math
\text{output}
= \sum_i \operatorname{softmax}(q^\top k_i)v_i
```

Suppose the previous values are scalar values:

```math
v_1 = 10,
\quad
v_2 = 50,
\quad
v_3 = 20
```

And suppose the attention weights are:

```math
\begin{bmatrix}
0.1 & 0.8 & 0.1
\end{bmatrix}
```

Then the Transformer output is:

```math
\text{output}
= 0.1(10) + 0.8(50) + 0.1(20)
```

So:

```math
\text{output}
= 1 + 40 + 2
= 43
```

The important thing is that the Transformer still has the separate values available:

```math
10,\quad 50,\quad 20
```

The attention weights decide how much to read from each previous token.

Gated DeltaNet is closer to:

```math
\text{output} \approx S_t q_t
```

It does not keep asking for a weighted mixture over explicit old tokens. The old tokens have already been folded into $S_t$. The current query/key reads whatever the compressed memory matrix returns.

## Why compressed memory can interfere

Use a scalar-value version of the memory, where $S$ is a row vector and the keys are column vectors. Reading is:

```math
\text{read}(k) = S k
```

Start with:

```math
S_0 =
\begin{bmatrix}
0 &
0
\end{bmatrix}
```

Store the first key-value pair:

```math
k_1 =
\begin{bmatrix}
1 \\
0
\end{bmatrix},
\quad
v_1 = 10
```

With $\alpha = 1$ and $\beta = 1$, the scalar-value delta update is:

```math
S_{\text{new}}
= S + (v - S k)k^\top
```

The current read is:

```math
S_0 k_1
=
\begin{bmatrix}
0 & 0
\end{bmatrix}
\begin{bmatrix}
1 \\
0
\end{bmatrix}
= 0
```

So the correction is:

```math
v_1 - S_0 k_1 = 10 - 0 = 10
```

Write it into the memory:

```math
S_1
= S_0 + 10k_1^\top
=
\begin{bmatrix}
0 &
0
\end{bmatrix}
+
10
\begin{bmatrix}
1 &
0
\end{bmatrix}
=
\begin{bmatrix}
10 &
0
\end{bmatrix}
```

Now store a second key-value pair:

```math
k_2 =
\begin{bmatrix}
1 \\
1
\end{bmatrix},
\quad
v_2 = 50
```

Before writing, read what memory already returns for $k_2$:

```math
S_1 k_2
=
\begin{bmatrix}
10 & 0
\end{bmatrix}
\begin{bmatrix}
1 \\
1
\end{bmatrix}
= 10
```

So the correction is:

```math
v_2 - S_1 k_2 = 50 - 10 = 40
```

Write that correction in the $k_2^\top$ direction:

```math
S_2
= S_1 + 40k_2^\top
=
\begin{bmatrix}
10 &
0
\end{bmatrix}
+
40
\begin{bmatrix}
1 &
1
\end{bmatrix}
=
\begin{bmatrix}
50 &
40
\end{bmatrix}
```

Now read the first key again:

```math
S_2 k_1
=
\begin{bmatrix}
50 & 40
\end{bmatrix}
\begin{bmatrix}
1 \\
0
\end{bmatrix}
= 50
```

The first value used to read as $10$, but now it reads as $50$.

That is interference. The second key was not separate from the first key. Since:

```math
k_1^\top k_2
=
\begin{bmatrix}
1 & 0
\end{bmatrix}
\begin{bmatrix}
1 \\
1
\end{bmatrix}
= 1
```

the second update also changed the direction used by the first key.

In a Transformer, $v_1 = 10$ and $v_2 = 50$ are still separate cached values. In Gated DeltaNet, both writes are folded into the same compressed memory, so overlapping keys can collide.

The property we want is orthogonal keys:

```math
k_1^\top k_2 = 0
```

Usually we also want normalized keys, so each key has length $1$:

```math
\lVert k_i \rVert = 1
```

For example:

```math
k_1 =
\begin{bmatrix}
1 \\
0
\end{bmatrix},
\quad
k_2 =
\begin{bmatrix}
0 \\
1
\end{bmatrix}
```

Then:

```math
k_1^\top k_2
=
\begin{bmatrix}
1 & 0
\end{bmatrix}
\begin{bmatrix}
0 \\
1
\end{bmatrix}
= 0
```

If one key is written, it does not activate the other key. So Gated DeltaNet works best when keys are separate. If keys overlap, memories can interfere.

Now redo the same two writes with orthogonal keys.

Start with:

```math
S_0 =
\begin{bmatrix}
0 & 0
\end{bmatrix}
```

Write the first value:

```math
k_1 =
\begin{bmatrix}
1 \\
0
\end{bmatrix},
\quad
v_1 = 10
```

The current read is:

```math
S_0 k_1
=
\begin{bmatrix}
0 & 0
\end{bmatrix}
\begin{bmatrix}
1 \\
0
\end{bmatrix}
= 0
```

So the correction is:

```math
v_1 - S_0 k_1 = 10 - 0 = 10
```

Write it:

```math
S_1
= S_0 + 10k_1^\top
=
\begin{bmatrix}
0 & 0
\end{bmatrix}
+
10
\begin{bmatrix}
1 & 0
\end{bmatrix}
=
\begin{bmatrix}
10 & 0
\end{bmatrix}
```

Now write the second value with the orthogonal key:

```math
k_2 =
\begin{bmatrix}
0 \\
1
\end{bmatrix},
\quad
v_2 = 50
```

Before writing, read what memory returns for $k_2$:

```math
S_1 k_2
=
\begin{bmatrix}
10 & 0
\end{bmatrix}
\begin{bmatrix}
0 \\
1
\end{bmatrix}
= 0
```

The first write does not show up when reading $k_2$, because:

```math
k_1^\top k_2 = 0
```

So the correction is:

```math
v_2 - S_1 k_2 = 50 - 0 = 50
```

Write it:

```math
S_2
= S_1 + 50k_2^\top
=
\begin{bmatrix}
10 & 0
\end{bmatrix}
+
50
\begin{bmatrix}
0 & 1
\end{bmatrix}
=
\begin{bmatrix}
10 & 50
\end{bmatrix}
```

Now read the first key again:

```math
S_2 k_1
=
\begin{bmatrix}
10 & 50
\end{bmatrix}
\begin{bmatrix}
1 \\
0
\end{bmatrix}
= 10
```

And read the second key:

```math
S_2 k_2
=
\begin{bmatrix}
10 & 50
\end{bmatrix}
\begin{bmatrix}
0 \\
1
\end{bmatrix}
= 50
```

So with orthogonal normalized keys, both memories stay clean:

```math
k_1 \rightarrow 10,
\quad
k_2 \rightarrow 50
```

In a real language model with thousands of tokens, keys are only approximately separate. That means Gated DeltaNet memory is always a bit lossy:

```math
\text{read from } k_2
\approx
\text{desired value}
+ \text{leakage from similar keys}
```

That leakage is the price of compression.

The comparison becomes:

| Model | Memory | Retrieval issue |
| --- | --- | --- |
| Transformer | Separate past $K,V$ vectors | Expensive, but direct |
| Mamba | Recurrent hidden state | No explicit key-value lookup |
| Gated DeltaNet | Compressed associative memory $S$ | Similar keys can interfere |

Gated DeltaNet tries to manage this with:

- $\alpha$: forget old state memory.
- $\beta$: control how aggressively to update a key-value association.

Now suppose the keys are normalized but not perfectly orthogonal:

```math
k_1^\top k_2 = 0.2
```

And suppose we already stored:

```math
k_1 \rightarrow 10
```

That means:

```math
S = 10k_1^\top
```

Now read from $k_2$:

```math
S k_2
= 10k_1^\top k_2
```

Since $k_1^\top k_2 = 0.2$:

```math
S k_2
= 10(0.2)
= 2
```

So even before writing anything to $k_2$, the memory already returns $2$ there.

That is leakage from the similar key:

```math
k_2 \text{ reads } 2 \text{ because it partially overlaps with } k_1
```

If the target for $k_2$ is $50$, the correction is not $50$. It is:

```math
50 - 2 = 48
```

The model only writes the missing part, because the memory already predicts $2$ at that key. This is useful, but also risky: similar keys share memory.

After the correction, the new memory is:

```math
S_{\text{new}}
= 10k_1^\top + 48k_2^\top
```

Now read from $k_2$:

```math
S_{\text{new}}k_2
= 10k_1^\top k_2 + 48k_2^\top k_2
```

Because the keys are normalized:

```math
k_2^\top k_2 = 1
```

So:

```math
S_{\text{new}}k_2
= 10(0.2) + 48(1)
= 2 + 48
= 50
```

The delta rule fixed the read for $k_2$.

But now read from $k_1$ again:

```math
S_{\text{new}}k_1
= 10k_1^\top k_1 + 48k_2^\top k_1
```

Since:

```math
k_1^\top k_1 = 1,
\quad
k_2^\top k_1 = 0.2
```

we get:

```math
S_{\text{new}}k_1
= 10(1) + 48(0.2)
= 10 + 9.6
= 19.6
```

So the delta rule corrected the current key, but damaged a similar key. That is the tradeoff: if keys overlap, the correction leaks back into nearby memories.

Now suppose $\beta = 0.5$ instead of $\beta = 1$.

The memory still currently reads $2$ from $k_2$:

```math
S k_2 = 2
```

The full correction would still be:

```math
50 - 2 = 48
```

But with $\beta = 0.5$, we only apply half of that correction:

```math
0.5(48) = 24
```

So the new memory is:

```math
S_{\text{new}}
= 10k_1^\top + 24k_2^\top
```

Now read from $k_2$:

```math
S_{\text{new}}k_2
= 10k_1^\top k_2 + 24k_2^\top k_2
```

So:

```math
S_{\text{new}}k_2
= 10(0.2) + 24(1)
= 2 + 24
= 26
```

The read for $k_2$ moved toward $50$, but did not fully reach it:

```math
2 \rightarrow 26
```

Now read from $k_1$ again:

```math
S_{\text{new}}k_1
= 10k_1^\top k_1 + 24k_2^\top k_1
```

So:

```math
S_{\text{new}}k_1
= 10(1) + 24(0.2)
= 10 + 4.8
= 14.8
```

So $\beta = 0.5$ makes the current edit weaker, but also reduces the side effect:

```math
k_2: 2 \rightarrow 26
```

```math
k_1: 10 \rightarrow 14.8
```

That is why $\beta$ matters. A larger $\beta$ fixes the current key more aggressively, but leaks more into similar keys. A smaller $\beta$ edits more gently, so the target key improves more slowly, but nearby memories are damaged less.

Now bring back $\alpha$.

Suppose the old memory is:

```math
S = 10k_1^\top
```

Before writing, apply a decay gate:

```math
\alpha = 0.8
```

The gated delta update is:

```math
S_{\text{new}}
= \alpha S + \beta(v - \alpha S k)k^\top
```

So the old $k_1$ memory is first decayed:

```math
\alpha S
= 0.8(10k_1^\top)
= 8k_1^\top
```

Reading from $k_1$ after decay gives:

```math
\alpha S k_1
= 8k_1^\top k_1
= 8(1)
= 8
```

So $\alpha$ turns the old memory:

```math
10 \rightarrow 8
```

If we now read from the similar key $k_2$, where $k_1^\top k_2 = 0.2$:

```math
\alpha S k_2
= 8k_1^\top k_2
= 8(0.2)
= 1.6
```

So decay also reduces leakage:

```math
2 \rightarrow 1.6
```

If the target for $k_2$ is still $50$, the correction becomes:

```math
50 - 1.6 = 48.4
```

So $\alpha$ forgets old memory before the edit happens. It reduces stale information and leakage, but it also weakens useful old memories.

With $\beta = 1$, the final memory after writing $k_2$ is:

```math
S_{\text{new}}
= 8k_1^\top + 48.4k_2^\top
```

Now read from $k_1$ again:

```math
S_{\text{new}}k_1
= 8k_1^\top k_1 + 48.4k_2^\top k_1
```

Since $k_1^\top k_1 = 1$ and $k_2^\top k_1 = 0.2$:

```math
S_{\text{new}}k_1
= 8(1) + 48.4(0.2)
= 8 + 9.68
= 17.68
```

In the no-decay case, reading from $k_1$ became:

```math
19.6
```

With $\alpha = 0.8$, reading from $k_1$ becomes:

```math
17.68
```

So $\alpha = 0.8$ reduced the final corruption compared with no decay:

```math
19.6 \rightarrow 17.68
```

But it did that partly by weakening the original $k_1$ memory first:

```math
10 \rightarrow 8
```

The main intuition is:

- $\alpha$ can reduce old memory before writing.
- $\beta$ can reduce how aggressively the model writes the correction.
- But if keys overlap, interference can still happen.

The delta rule turns the write into a correction:

```math
\text{write correction}
= \beta(v - \text{current read})
```

where:

```math
\text{current read} = S k
```

So:

```math
\text{write correction}
= \beta(v - S k)
```

If the memory already retrieves the right value, then:

```math
S k \approx v
```

so:

```math
v - S k \approx 0
```

and the model writes almost nothing.

The caveat is that this is not always a perfect replacement. If $\alpha < 1$, old memory is decayed before the write. If $\beta < 1$, the correction is only partially written. And if keys overlap, the correction for one key can interfere with nearby keys.

# Comparing Gated DeltaNet to Mamba

A simplified Mamba-style recurrence looks like:

```math
h_t = A_t h_{t-1} + B_t x_t
```

Then the output is read from the hidden state:

```math
y_t = C_t h_t
```

So Mamba also keeps a compressed state instead of storing every previous token separately.

But the update is a state-space recurrence. It says:

```text
How should I update my running state?
```

Gated DeltaNet has a different flavor. Its update is more like:

```math
S_t
= \text{old memory}
+ \text{key-specific correction}
```

More explicitly:

```math
S_t
= \alpha_t S_{t-1}
+ \beta_t \left(v_t - \alpha_t S_{t-1} k_t\right) k_t^\top
```

That means it reads what is currently stored at key $k_t$, compares it to the target value $v_t$, and edits the memory matrix using the correction.

So Gated DeltaNet is asking:

```text
How should I edit this key-value memory?
```

The rough picture is:

- Transformer: explicit key/value memory for previous tokens.
- Mamba: compressed running hidden state.
- Gated DeltaNet: compressed editable key-value memory.

So Gated DeltaNet sits between Transformer and Mamba. It is compressed like Mamba, but its update has a key-value edit operation that feels closer to attention memory.
