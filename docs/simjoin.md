# Overview
Supported similarity join algorithms: Jaccard, Cosine, Dice, Overlap, Edit-distance. They are from three papers.

- Dong Deng, Yufei Tao, Guoliang Li: Overlap Set Similarity Joins with Theoretical Guarantees. SIGMOD Conference 2018: 905-920 

- Dong Deng, Guoliang Li, He Wen, Jianhua Feng: An Efficient Partition Based Method for Exact Set Similarity Joins. Proc. VLDB Endow. 9(4): 360-371 (2015) 

- Guoliang Li, Dong Deng, Jiannan Wang, Jianhua Feng: PASS-JOIN: A Partition-based Method for Similarity Joins. Proc.of the VLDB Endow. 5(3): 253-264 (2011)

# Appendix

## A. Adaptive Grouping in Set Join

We have $H(\frac{l}{\alpha},s) + 1 \leq 2 (H_{l} + 1)$, where $H(\frac{l}{\alpha}, s) = \lfloor \frac{1 - \delta}{1 + \delta} (s + \frac{l}{\alpha}) \rfloor$ for Jaccard and $H(\frac{l}{\alpha}, s) = \lfloor s + \frac{l}{\alpha} - 2\delta\sqrt{s\frac{l}{\alpha}} \rfloor$ fpr Cosine. And $H_l = \lfloor \frac{1 - \delta}{\delta}l \rfloor$ for Jaccard and $H_l = \lfloor \frac{1 - \delta^2}{\delta^2}l \rfloor$ for Cosine.

### Jaccard

That is, $\lfloor \frac{1 - \delta}{1 + \delta} (s + \frac{l}{\alpha}) \rfloor + 1 < 2 (\lfloor\frac{1 - \delta}{\delta}l \rfloor + 1)$

Set $L = \lfloor \frac{1 - \delta}{1 + \delta} (s + \frac{l}{\alpha}) \rfloor$ and $R = \lfloor\frac{1 - \delta}{\delta}l \rfloor$

Rewrite the above inequality: $L + 1 \leq 2(R + 1)$

For $L$, we have $\lfloor \frac{1 - \delta}{1 + \delta} (s + \frac{l}{\alpha}) \rfloor \leq \frac{1 - \delta}{1 + \delta} (s + \frac{l}{\alpha}) < \lfloor \frac{1 - \delta}{1 + \delta} (s + \frac{l}{\alpha}) \rfloor + 1$

For $R$, we have $\lfloor\frac{1 - \delta}{\delta}l \rfloor \leq \frac{1 - \delta}{\delta}l < \lfloor\frac{1 - \delta}{\delta}l \rfloor + 1$

We aim to set $L_{max} + 1 < 2(R_{min} + 1)$

That is, $\frac{1 - \delta}{1 + \delta} (s + \frac{l}{\alpha}) + 1 < 2 \frac{1 - \delta}{\delta}l$

Simplify by $l \leq s \leq \frac{l}{\alpha\delta}$, we get $\frac{1 - \delta}{\delta} \frac{l}{\alpha} + 1 < 2 \frac{1 - \delta}{\delta}l$

Put $l$ with $\alpha$ on the same side, $\frac{1-\delta}{\delta}l(\frac{1}{\alpha} - 2) < -1$

Thus, $\frac{1}{\alpha} < 2 - \frac{\delta}{l(1 - \delta)}$

### Cosine
Similar as the calculation of Jaccard, we firstly have $\lfloor s + \frac{l}{\alpha} - 2\delta\sqrt{s\frac{l}{\alpha}} \rfloor + 1 \leq 2 (\lfloor \frac{1 - \delta^2}{\delta^2}l\rfloor + 1)$

set $L = \lfloor s + \frac{l}{\alpha} - 2\delta\sqrt{s\frac{l}{\alpha}} \rfloor$, we have $\lfloor s + \frac{l}{\alpha} - 2\delta\sqrt{s\frac{l}{\alpha}} \rfloor \leq s + \frac{l}{\alpha} - 2\delta\sqrt{s\frac{l}{\alpha}} < \lfloor s + \frac{l}{\alpha} - 2\delta\sqrt{s\frac{l}{\alpha}} \rfloor + 1$

set $R = \lfloor \frac{1 - \delta^2}{\delta^2}\rfloor$, we have $\lfloor \frac{1 - \delta^2}{\delta^2}l\rfloor \leq \frac{1 - \delta^2}{\delta^2}l < \lfloor \frac{1 - \delta^2}{\delta^2}l\rfloor + 1$

Then we deduce $s + \frac{l}{\alpha} - 2\delta\sqrt{s\frac{l}{\alpha}} + 1 < 2\frac{1 - \delta^2}{\delta^2}l$

Simplify by $l \leq s \leq \frac{l}{\alpha\delta^2}$, it is easy to find that the left hand side is $\sqrt{s}(\sqrt{s} - 2\delta\sqrt{\frac{l}{\alpha}}) + \frac{l}{\alpha}$. Hence, to deduce the $L_{max}$, we rewrite $(\sqrt{s} - \delta\sqrt{\frac{l}{\alpha}})^2 + (1 - \delta^2)\frac{l}{\alpha}$, it's trivial that $\sqrt{s} = \delta\sqrt{\frac{l}{\alpha}}$ we have $L_{min}$. Hence, we have max value when $s = l$, that is $L_{max} = l(\frac{1}{\alpha} + 1 - 2\delta\sqrt{\frac{1}{\alpha}})$.

Hence, we deduce: $l(\frac{1}{\alpha} + 1 - 2\delta\sqrt{\frac{1}{\alpha}}) < 2 \frac{1 - \delta^2}{\delta^2}l$, which can not be simplified to keep $\alpha$