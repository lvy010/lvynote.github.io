# lec6 Managed Memory | 异构编程

### CUDA 6 + 时代的统一内存（Unified Memory）

让 Kepler 架构 ==GPU 和 CPU 能通过单块内存区域协同工作==，既简化了编程模型（单分配、单指针、无需显式拷贝），又通过数据本地化保证性能，大幅降低了开发者在异构内存管理上的工作量。

![image-20251121173543099](image-20251121173543099.png)

CUDA 8 + 的统一内存技术支持 Pascal 及后续架构 GPU，==可超量使用 GPU 内存===（分配规模突破 GPU 自身显存限制），简化 CPU/GPU 数据访问与一致性管理，还能通过 API 调优性能，助力开发者高效处理大规模数据模型。

![image-20251121173724885](image-20251121173724885.png)

> 实现超量

系统==会自动把多余的内存页 “挪” 到 CPU 内存里==；要是想进一步优化，还可以用`cudaMemAdvise`这类 API 给内存访问 “提建议”，或者提前把数据 “搬” 到要用的设备上，这样能让性能更好些。

 CUDA 统一内存技术如何简化内存管理

- 左侧 CPU 代码无需显式处理 GPU 内存分配、拷贝

- 右侧传统 CUDA 代码需手动完成的内存操作（分配、拷贝、释放）被大幅简化，让开发者编写异构内存程序更高效。

（这张图算是cuda梦开始的地方了，cuda底层启发的感知机（雾

![image-20251121173825649](image-20251121173825649.png)

`qsort` 是 C 语言标准库的快速排序 API，能对任意类型数组按自定义比较规则进行高效排序，是灵活且实用的通用排序工具。

关于qsort，大一学C语言梦开始的地方，那时感觉脑子还不是很好，一点知识就要学挺久...但还是挺好玩的，回看感觉也挺神奇的，传送：[15.C与指针--超全总结](https://blog.csdn.net/2301_80171004/article/details/136834259?ops_request_misc=%257B%2522request%255Fid%2522%253A%25222ac4d7ab5e7508ebde1bab72b19ac85a%2522%252C%2522scm%2522%253A%252220140713.130102334.pc%255Fblog.%2522%257D&request_id=2ac4d7ab5e7508ebde1bab72b19ac85a&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~blog~first_rank_ecpm_v1~rank_v31_ecpm-2-136834259-null-null.nonecase&utm_term=qsort&spm=1018.2226.3001.4450)

### CUDA 统一内存（`cudaMallocManaged`）

上面内存切来切去是不是很麻烦，我们来统一管理起来叭😋

![image-20251121174449292](image-20251121174449292.png)

CUDA 统一内存（`cudaMallocManaged`）让同一块内存可在 CPU 和 GPU 代码中等效使用，==结合按需分页机制==，开发者==无需手动==处理内存跨设备访问，大幅简化异构编程的内存管理。

### Interconnect（互联）

在 Pascal + 架构上，CUDA 统一内存通过 Interconnect（互联）机制，为 GPU 和 CPU 分别==维护内存映射==，自动处理跨设备的页故障，实现 CPU 与 GPU 对同一块内存的协同访问。

![image-20251121174538151](image-20251121174538151.png)

Pascal 之前架构（或 Windows 下 CUDA 9.x+）的统一内存，在核函数启动时批量迁移数据，无并发访问和按需迁移，需用`cudaMallocManaged`替代`malloc`、`cudaFree`替代`free`，且核函数同步后数据才对 CPU 可见。

Pascal + 架构的 CUDA 统一内存支持超量使用 GPU 内存（如 16GB GPU 可分配 64GB），仅按需迁移部分内存页到 GPU，而 Kepler/Maxwell 架构不支持此特性。

![image-20251121174632949](image-20251121174632949.png)

> 16GB GPU可分配64GB

GPU 的显存就像它的 “小仓库”，本来只能装 16GB 东西。但==有了统一内存技术==，它==可以把暂时不用的东西先放到 CPU== 的 “大仓库”（系统内存）里，要用的时候再拿回来。

这样整体能调配的内存就不止 16GB 了，甚至能达到 64GB，就像==借了 CPU 的内存来扩展自己的可用空间==

![image-20251121174802592](image-20251121174802592.png)

Pascal + 架构的 CUDA 统一内存支持系统级原子操作，==GPU 和 CPU 可通过 NVLink 或 PCIe 协同执行原子指令==，而 Kepler/Maxwell 架构不具备此特性。

### CUDA 中深拷贝

![image-20251121174906634](image-20251121174906634.png)

需==同时传输结构体对象及其内部缓冲区到设备==，并修正设备端指针使其指向缓冲区的新地址，通过==手动分配==、拷贝内存来实现这一过程。

![image-20251121174948191](image-20251121174948191.png)

CUDA 统一内存让链表这类需深拷贝的复杂数据结构在异构编程中变得极易处理，原本繁琐的拷贝操作因统一内存技术变得简单。直接传地址即可。

对于深浅拷贝前文传送：

[[C++11#45](二) 右值引用 | 移动语义 | 万能引用 | 完美转发forward | 初识lambda](https://blog.csdn.net/2301_80171004/article/details/141871371?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522211c2a0c1a557b7df57076f64837ae04%2522%252C%2522scm%2522%253A%252220140713.130102334.pc%255Fblog.%2522%257D&request_id=211c2a0c1a557b7df57076f64837ae04&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~blog~first_rank_ecpm_v1~rank_v31_ecpm-1-141871371-null-null.nonecase&utm_term=%E5%8F%B3%E5%80%BC%E5%BC%95%E7%94%A8&spm=1018.2226.3001.4450)

[【C++】手动模拟String底层与深浅拷贝](https://blog.csdn.net/2301_80171004/article/details/139445375?ops_request_misc=%257B%2522request%255Fid%2522%253A%25223f7639690beb71d3c8791362e2e6f516%2522%252C%2522scm%2522%253A%252220140713.130102334.pc%255Fblog.%2522%257D&request_id=3f7639690beb71d3c8791362e2e6f516&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~blog~first_rank_ecpm_v1~rank_v31_ecpm-2-139445375-null-null.nonecase&utm_term=%E6%B7%B1%E6%B5%85%E6%8B%B7%E8%B4%9D&spm=1018.2226.3001.4450)

第一层接触到应该是在看string底层的时候，那时候真的很喜欢画导图desu

----

### 给 C++ 字符串开了个‘共享内存 buff

在 C++ 基类中重载`new`和`delete`操作符（基于 CUDA 统一内存的`cudaMallocManaged`和`cudaFree`），可让派生类（如字符串类）在异构编程中便捷地管理内存，实现 CPU 与 GPU 间的对象共享。

![image-20251121175436074](image-20251121175436074.png)

继承重载了内存管理的基类，字符串对象就能在 CPU 和 GPU 之间自动共享内存，不用你手动折腾数据拷贝，写代码时像普通 C++ 对象一样用就行～

![image-20251121175510225](image-20251121175510225.png)

CUDA 统一内存的按需分页机制在图算法场景中，可==根据已知访问模式（如顺序访问）智能管理内存页迁移==，助力高效处理图结构的异构计算任务。

![image-20251121175654430](image-20251121175654430.png)

