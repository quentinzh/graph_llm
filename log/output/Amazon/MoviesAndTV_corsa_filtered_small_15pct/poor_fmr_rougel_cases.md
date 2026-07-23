# graph_llm：FMR 与 ROUGE-L 均较差的测试样例

来源：`graph_llm/log/output/Amazon/MoviesAndTV_corsa_filtered_small_15pct/1generate.all.dataset`，共 4299 条完整测试输出。
筛选规则：逐样本 **FMR=0**，再按 **ROUGE-L F1** 升序取前 10 条。

| # | 测试集位置 | 原始数据行 | Feature | FMR | ROUGE-L | graph_llm 输出 | 测试集真实结果 |
| --- | ---: | ---: | --- | ---: | ---: | --- | --- |
| 1 | 9 | 38705 | career | 0 | 0.00 | the movie is less than 100 minutes long | nic cage 's career has been one heck of a roller-coaster ride |
| 2 | 18 | 38714 | trademark | 0 | 0.00 | the film is a romantic comedy | slow-building 1948 thriller replete with his trademark technical precision |
| 3 | 23 | 38719 | movie | 0 | 0.00 | the octopus is a great monster | neat vintage creature movie |
| 4 | 33 | 38729 | movies | 0 | 0.00 | the transfer is older and the picture is grainy | no wonder very few movies are this good |
| 5 | 38 | 38734 | cast | 0 | 0.00 | the film is a great adaptation of the novel and the performances are excellent | no cast info |
| 6 | 51 | 38747 | spots | 0 | 0.00 | the sound is low and the picture is low resolution | key spots include hawaii |
| 7 | 55 | 38751 | extras | 0 | 0.00 | i give this movie 5 stars | the extras are pretty interesting and you get to see the process of restoring ht efilm and bringing it back to life from the faded negatives |
| 8 | 56 | 38752 | fans | 0 | 0.00 | the film is a bit slow in the beginning | i would recommend this to fans of hard scifi |
| 9 | 61 | 38757 | moments | 0 | 0.00 | nice visuals | from that we are treated with some scary and creepy moments of ghostly raptures |
| 10 | 66 | 38762 | movie | 0 | 0.00 | the audio is crisp and clean | if you want to see a great movie about a virus |
