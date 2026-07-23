# graph_llm：相对容易但输出错误的测试样例

筛选依据：目标 feature 在训练集中的出现频率较高、对应物品在训练集中有较多交互、真实解释较短且表达通用；同时要求模型输出未命中目标 feature（FMR=0），并且 ROUGE-L 很低。

| # | 测试集位置 | 原始数据行 | Feature | Feature 训练频次 | Item 训练频次 | graph_llm 输出 | 测试集真实结果 |
| --- | ---: | ---: | --- | ---: | ---: | --- | --- |
| 1 | 1393 | 40089 | movie | 4790 | 41 | the phantom menace is a rough film | i was just like everyone else--i wanted to see this movie really really badly |
| 2 | 3679 | 42375 | movie | 4790 | 40 | the cinematography is stunning and the special effects are amazing | this movie far outshone avatar |
| 3 | 693 | 39389 | movie | 4790 | 31 | the plot is pretty good | it 's a very nice movie to see |
| 4 | 1395 | 40091 | movie | 4790 | 27 | the ending is one of the most shocking moments in cinema history | it was a good movie till it ended |
| 5 | 4092 | 42788 | movie | 4790 | 26 | the special features are also quite good | a definite must see for any movie lover like me |
| 6 | 468 | 39164 | movie | 4790 | 23 | the zombies are a bit too human-like in their movements and behavior | this movie was excellent |
| 7 | 1114 | 39810 | movie | 4790 | 21 | the super hero is the one who saves the day | i liked this movie |
| 8 | 2131 | 40827 | movie | 4790 | 20 | the plot is simple enough to follow | strange movie but a great view |
| 9 | 3693 | 42389 | movie | 4790 | 17 | the high budget | people say this movie is funny but i did not see it |
| 10 | 3081 | 41777 | movie | 4790 | 19 | i liked the humor in the film | but this was a poorly made movie |

