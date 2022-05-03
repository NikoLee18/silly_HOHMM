这个repo里放的是我毕业论文写的高阶隐马尔可夫模型（HOHMM）的东西。 我参考了Hardar和Messer等人（特拉维夫大学）将HOHMM转换为一个参数受限+有冗余状态表示的一阶HMM的方法。但是由于Hardar论文中的表示方法
不太适合数值计算，所以我做了一点小修改，具体内容在我的论文中，答辩完了会发上来。 也会更新到我的微信公众号和知乎专栏里。  或许我闲下来的时候还可以专门做一个b站视频来讲解。

目前把挑战最大的E-step写完了，数值稳定，使用的不是Forward-Backward(alpha-beta)方法，而是动态线性系统用的Alpha-Gamma方法（请参考PRML中HMM的scaling factor那一部分）。

如果大家对怎样把原始的Alpha-beta方法写成向量计算的形式，可以看一下model.py里SillyHMM类。 稳定的请参考HMM类。
