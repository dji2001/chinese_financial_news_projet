Project : Chinese financial news Kmeans++ clustering based on TFIDF vectorization
data source: https://figshare.com/articles/dataset/Chinese_Financial_News_Data/12666233

1. Required library and instruction to run:
    Zhon: for Chinese punctuation
    Jieba: for Chinese language segmentation
    Other regular: matplotlib, sklearn, numpy, pandas, etc

    data saved in article directory
    code and results in src directory

    Instructions: I didn't design this to take arguments, conversion should be simple, but I am runing out of time
                  besides, runing everthing takes FOREVER. 
                  
                  Nevertheless, if you are still interested:
                  
                  segmentation.py: modify segmentation(dirname,output_name) in the bottom
                                   dirname refers to the directory of the database, 
                                   note you have to put texts in dirname/Some_other_dir/texts
                                   output will be saved in dirname/output_name
                 
                  TF_IDF_vectorize.py: find driver() at the bottom
                                  TFIDF=Tf_Idf(dirname,filename,sample_num)
                                  first two arguments being the two you used for segmentation
                                  third argument being the sample_size in integer
                  
                  K_meansPP.py: find main() below and input a list of integers as k_vals
                                e.g. main([2,3,5,8,10])

2. segmentation
    Database includes: 18956 chinese financial news article in total, grouped in 2088 folders by news date
    article example:
    商务部新闻发言人沈丹阳17日在例行发布会上表示，今年外贸形势总体上还是比较严峻，但是我们仍然非常有信心来实现今年既定的外贸发展目标。而实现这一目标，首先要在“稳增长”方面下功夫。
    《证券日报》报道，沈丹阳表示，今年一季度外贸进出口增长了7.3%，这样一个相对低速的增长是在多种因素相互叠加的作用下出现的。
    沈丹阳表示，外贸发展面临的国内外环境仍然复杂严峻，形势仍然不容乐观。目前商务部和各级商务部门都已经积极行动起来，相信随着形势的发展和工作的深入，二季度外贸形势还会进一步好转，全年我们既定的外贸目标应该可以实现。全年外贸发展的主要目标是三句话九个字，即“稳增长、调结构、促平衡”，首先要在“稳增长”方面下功夫：
    第一是稳定政策。出口要稳定，首先必须政策稳定。企业最关心的出口退税政策、出口信用保险政策、贸易融资等金融支持政策，不仅要稳定，而且要加大支持的力度。国务院领导和商务部领导也多次强调，政策如果要做适度微调，也是鼓励性多于限制性的。
    所有具有进出口经营资格的企业都可以开展出口货物贸易项下的人民币结算业务，这个措施对于支持外贸有效应对人民币汇率波动的影响是有利的。沈丹阳表示，“虽然人民币汇率扩大交易的波动，企业有担心，这个办法就可以帮助企业了。此外，我们还会加快出口退税进度，加大对成套设备出口的信用保险支持，清理各种不合理的收费，有针对性地解决重点行业和中小企业的实际困难等。”
    第二是加强引导。要引导企业深度开拓新兴市场，引导企业在国外建立营销网络，引导企业培育自主品牌，引导加工贸易向产业链高端延伸，向中西部转移。
    第三是优化服务，在推动提高贸易便利化水平的同时，我们将对重点地区加强指导，对重点行业和企业加强服务。
    在“调结构”方面，沈丹阳表示，“我们实际上已经做了很多工作，去年取得了很好的成效，今年继续做。”
    “在“促平衡”方面，上个月国务院常务会议已经通过了一个文件，这个文件即将下发，题目叫做《关于加强进口促进对外贸易平衡发展的指导意见》，这个意见里有很多具体的政策措施，对今年进一步扩大进口、促进平衡将会有很好的指导作用。”沈丹阳表示。

    segmentation.py: take all articles in article folder and segment with jieba
                 save results in article/segmented.txt
                 Note: I haven't add customrized dictionary, but it is beyond the scope of this project

    segmented example:
    商务部 新闻 发言人 沈 丹阳 17 日 在 例行 发布会 上 表示 今年 外贸 形势 总体 上 还是 比较 严峻 但是 我们 仍然 非常 有 信心 来 实现 今年 既定 的 外贸 发展 目标 而 实现 这一 目标 首先 要 在 稳 增长 方面 下功夫 证券日报 报道 沈 丹阳 表示 今年 一季度 外贸 进出口 增长 了 7.3% 这样 一个 相对 低速 的 增长 是 在 多种 因素 相互 叠加 的 作用 下 出现 的 沈 丹阳 表示 外贸 发展 面临 的 国内外 环境 仍然 复杂 严峻 形势 仍然 不容乐观 目前 商务部 和 各级 商务 部门 都 已经 积极行动 起来 相信 随着 形势 的 发展 和 工作 的 深入 二季度 外贸 形势 还会 进一步 好转 全年 我们 既定 的 外贸 目标 应该 可以 实现 全年 外贸 发展 的 主要 目标 是 三句话 九个 字 即 稳 增长 调 结构 促 平衡 首先 要 在 稳 增长 方面 下功夫 第一 是 稳定 政策 出口 要 稳定 首先 必须 政策 稳定 企业 最 关心 的 出口 退税 政策 出口 信用 保险 政策 贸易 融资 等 金融 支持 政策 不仅 要 稳定 而且 要 加大 支持 的 力度 国务院 领导 和 商务部 领导 也 多次 强调 政策 如果 要 做 适度 微调 也 是 鼓励性 多于 限制性 的 所有 具有 进出口 经营 资格 的 企业 都 可以 开展 出口 货物贸易 项下 的 人民币 结算 业务 这个 措施 对于 支持 外贸 有 效应 对 人民币 汇率 波动 的 影响 是 有利 的 沈 丹阳 表示 虽然 人民币 汇率 扩大 交易 的 波动 企业 有 担心 这个 办法 就 可以 帮助 企业 了 此外 我们 还会 加快 出口 退税 进度 加大 对 成套设备 出口 的 信用 保险 支持 清理 各种 不合理 的 收费 有 针对性 地 解决 重点 行业 和 中小企业 的 实际困难 等 第二 是 加强 引导 要 引导 企业 深度 开拓 新兴 市场 引导 企业 在 国外 建立 营销 网络 引导 企业 培育 自主 品牌 引导 加工 贸易 向 产业链 高端 延伸 向 中西部 转移 第三 是 优化 服务 在 推动 提高 贸易 便利化 水平 的 同时 我们 将 对 重点 地区 加强指导 对 重点 行业 和 企业 加强 服务 在 调 结构 方面 沈 丹阳 表示 我们 实际上 已经 做 了 很多 工作 去年 取得 了 很 好 的 成效 今年 继续 做 在 促 平衡 方面 上个月 国务院 常务会议 已经 通过 了 一个 文件 这个 文件 即将 下发 题目 叫做 关于 加强 进口 促进 对外贸易 平衡 发展 的 指导 意见 这个 意见 里 有 很多 具体 的 政策措施 对 今年 进一步 扩大 进口 促进 平衡 将会 有 很 好 的 指导作用 沈 丹阳 表示 

3. Code organization and implementation specifics

3A. TFIDF.py:
    take segmented texts from article/segmented.txt and run TF-IDF calculation
    IDF is calculated based on all 18959 arcticles, 
    TF-IDF score is only calculated  within sample size (manually set as 4000)
    and I added penalty to punish long articles, so the TFIDF score is calculated by:

        TFIDF[key]=(TF_score/article_length)*IDF_score

    I manually write the TF-IDF code because i will need to extract cluster-wise
    topics later and I have no idea how to do that with sklearn
    besides, I find dataframe VERY SLOW when processing data with very high dimensionality.
    (324 features after PCA reduction with 0.9 variance, so imagine the dimensionality)
    
    outputs:
    IDF.csv and TFIDF.csv outputed and saved in src directory, don't forget to
    NOTE: change encoding to utf8 bom beforing checking it out, if you are intersted.

3B. KmeansPP.py:
    the kmeans++ algorithm and visualization
    1. pca dimension reduce with 0.9 variance, 324 features after reduction
    
    2. just regular Kmeans++ algorithm with cosine distance
    
    2. I added topic_extract to extract significant topics within each group, calculated by
        simply add all TFIDF score of all articles with in one group up and multiply
        and sort them, print the 8 most significant words, this could be improved,
        as simply adding up all TFIDF score would benefit the common words that appear
        in all artciles, such as 的 and 和 (common Chinese words like "and" and "of")
    
    3. I manually selected 2,3,5,8,10 as K values

    4. outputs:
        f"fig_k={k}.png" are the visualization with PCA reduced to two features
        There is something wrong with the pictures, but honestly running it again takes too long
        I'll just leave it like this

        f"topics_simple_k={k}.txt" are the extracted topics followed by 20 article names in the cluster
        f"topics_comprehensive_k={k}.txt" are the extracted topics followed by all article names in the cluster
                                        not related to this project, but could be useful for future improvemen

4. clusters with keywords :
    As mentioned above, the topic extraction algorithm needs to be improved
    so its better to check the article names in topics_simple_k={k}.txt or to check next section
    but I do not have time for that, besides, I believe the current extracted
    topics still could give you a sense of the clusters:

    when k=2 topics:
        涨/拉升/涨超/逾/涨停/e/发稿/讯/
        的/人民币/元/企业/亿元/和/基金/对/

    when k=3 topics:
        涨/拉升/涨超/e/涨停/讯/逾/发稿/
        的/企业/基金/亿元/和/市场/家/在/
        人民币/元/1/环保/跌/水泥/对/逾/

    when k=5 topics:
        人民币/银行/央行/的/货币/汇率/解禁/区块/
        拉升/涨停/异动/直线/跟/概念股/午后/涨/
        的/企业/基金/和/亿元/家/市场/增长/
        逾/涨/4%/7%/跌/走强/5%/股份/
        涨超/e/讯/证券时报/元/人民币/发稿/涨停/

    when k-8 topics:
        ST/水泥/燃气/A/涨停/早盘/天然气/走强/
        的/点/发展/和/建设/经济/在/5G/
        拉升/涨/跟/异动/直线/证券/涨停/午后/
        人民币/元/1/汇率/对/中国外汇交易中心/美元/中间价/
        的/同比/增长/月份/净利润/指数/亿元/万元/
        涨超/e/讯/证券时报/银行/发稿/拉升/涨停/
        的/基金/投资者/企业/市场/发行/资本/上市公司/
        逾/涨/股份/5%/跌/科技/4%/走强/

    k=10 does not make much sense as it becomes to specific

5. Analysis with example k=8:
        
        cluster topics with manual editing and arcticle examples within:
        
    Cluster1:
        /涨停/早盘//走强/: 
        this is clearly about stock market going up（走强）and reaching limit up(涨停)

        2018-01-16/天然气概念股活跃走强贵州燃气涨停.txt
        2018-03-02/乡村振兴概念股午后活跃走强智慧农业涨5%.txt
        2017-08-18/雄安概念股快速走强博天环境涨停.txt
        2017-10-23/上海自贸区概念股再度走强华贸物流等多股涨停.txt
        2017-11-03/360概念股回落明显天业股份等相继开板.txt
        2018-10-10/次新股持续活跃锋龙股份5连板涨停.txt
        2017-12-01/网络安全概念股拉升走强拓尔思涨超8%.txt
        2017-09-26/可燃冰概念股快速走强潜能恒信涨超7%.txt
        2017-11-14/环保股活跃走强易世达涨超7%.txt
        2019-03-19/天然气板块午后拉升新疆浩源冲击涨停.txt
        2017-11-07/360概念股集体高开.txt
        2019-11-14/大基建板块持续走升北新路桥冲击涨停.txt
        2017-10-16/草甘膦概念股拉升走强长青股份大涨83%.txt
        2017-11-29/水泥股开盘走高上峰水泥涨停.txt
        2018-03-22/中石油拟新建八座储气库储气调峰行业补短板.txt
        2017-10-17/环保板块集体走低中环环保跌逾9%.txt
        2017-12-26/次新股表现抢眼新余国科等多股涨停.txt
        2020-06-08/OLED板块早盘集体强势涨370%京东方Ａ涨停.txt
        2018-04-04/5G板块持续活跃中光防雷等涨停.txt
        2018-01-08/区块链概念股再度走强新晨科技等2股涨停.txt
    
    Cluster2:
        发展/建设/经济//5G/
        This is clearly about macro Economic Situation Analysis

        2018-12-07/大势分析（2018年12月07日）贸.txt
        2018-03-12/大势分析（2018年03月13日）.txt
        2018-07-17/大势分析主力动向与投资机会（20.txt
        2017-12-05/专家预计2018年中国货币政策将延续紧平衡.txt
        2018-05-16/今日视点下阶段货币政策将施力结构性去杠杆.txt
        2020-04-01/大势分析（2020年04月01日）.txt
        2018-04-03/大势分析（2018年04月04日）向.txt
        2015-08-24/河北钢铁(000709)定增80亿布局高强度汽车板业务.txt
        2017-09-22/5G概念持续活跃关注产业链股投资机会.txt
        2018-02-23/大势分析（2018年02月26日）市.txt
        2019-11-14/多部委积极行动力挺海南自由贸易港建设.txt
        2017-09-25/大势分析2017年09月26日）3.txt
        2018-11-02/软件服务板块走势活跃托尔思科创信息等涨停.txt
        2018-09-19/加码基建与新兴业务中国中冶(601618)正焕发新生机.txt
        2014-03-25/发改委同意东莞等30城市建电子商务示范城市.txt
        2019-12-26/区块链信任机制推动普惠金融发展助力解决中小微企业融资难题.txt
        2020-03-10/大势分析（2020年03月11日）.txt
        2019-05-13/苏州升级版调控政策落地多城楼市或加码调控.txt
        2017-08-04/大势分析主力动向与投资机会（201.txt
        2018-01-30/大势分析（2018年01月31日）.txt
    
    Cluster3:
        拉升/涨/跟/异动/直线/证券/涨停/午后/
        Another cluster about stock market reaching up limit,well, k=8 might be too specific

        2020-03-19/券商股异动拉升中银证券涨停.txt
        2020-06-30/金融科技概念盘中拉升京天利、银之杰涨停.txt
        2017-10-25/汽车板块午后异动拉升天汽模涨逾6%.txt
        2018-02-28/券商股午后直线拉升山西证券涨超6%.txt
        2018-08-23/动物疫苗概念股异动瑞普生物涨停.txt
        2017-11-15/黄金概念股异动拉升园城黄金涨逾7%.txt
        2019-08-26/军工板块午后异动中国应急涨6%.txt
        2019-03-25/高送转概念午后拉升中光防雷直线涨停.txt
        2018-04-13/半导体板块午后走强阿石创封板.txt
        2019-05-14/广电系个股普遍拉升天威视讯、广电网络涨停.txt
        2019-07-16/钛金属概念强势天原集团涨停.txt
        2019-11-12/区块链概念股拉升当代东方拉升封板.txt
        2018-01-15/房地产板块午后异动拉升.txt
        2017-11-27/券商概念股午后快速拉升西部证券涨超4%.txt
        2020-04-28/农业股异动大康农业直线涨停.txt
        2017-12-12/有色金属板块走势活跃利源精制涨逾3%.txt
        2018-01-19/部分区块链概念股拉升科蓝软件直线涨停.txt
        2019-06-20/军工板块异动航天晨光直线涨停.txt
        2018-07-20/券商股午后发力沪指涨逾1%站上2800点.txt
        2019-01-03/券商股大幅走高方正证券率先涨停.txt
    
    Cluster4:
        人民币/元/1/汇率/对/中国外汇交易中心/美元/中间价/
        This is very clearly about exchange rate, especially US dollar and Chinese Yuan

        2014-11-11/[快讯]11月11日银行间外汇市场人民币汇率中间价.txt
        2017-09-08/8月份外汇储备环增108亿美元实现七连升至309万亿.txt
        2014-09-17/[快讯]9月17日银行间外汇市场人民币汇率中间价.txt
        2016-06-06/外汇流出压力放缓5月份外汇占款或环比持平.txt
        2014-10-20/10月20日银行间外汇市场人民币汇率中间价.txt
        2018-07-11/人民币对一篮子货币汇率昨日全线上涨.txt
        2015-02-11/[快讯]2月11日银行间外汇市场人民币汇率中间价.txt
        2016-08-05/人民币汇率改革一周年市场化成效初显.txt
        2015-02-03/人民币即期汇率再度逼近跌停折价幅度创记录.txt
        2013-02-19/油价上调预期升温液化气价格大涨几成定局.txt
        2014-07-30/[快讯]7月30日银行间外汇市场人民币汇率中间价.txt
        2013-04-23/人民币汇率波幅有望扩大走势均衡之时成最佳时点.txt
        2020-04-02/光伏概念股盘中拉升永高股份逼近涨停.txt
        2016-10-10/9月外储创新低人民币贬值压力或顺势释放.txt
        2015-02-09/[快讯]2月9日银行间外汇市场人民币汇率中间价.txt
        2014-03-19/人民币即期汇率破620或刺破房产泡沫拖累股市.txt
        2015-01-06/[快讯]1月6日银行间外汇市场人民币汇率中间价.txt
        2013-11-13/专家人民币汇率年底进五成大概率事件.txt
        2012-07-25/煤炭开采港口产地价续跌国际焦煤合同价反弹.txt
        2014-06-12/6月12日银行间外汇市场人民币汇率中间价.txt
    
    Cluster5:
        的/同比/增长/月份/净利润/指数/亿元/万元/
        This is about revenure report of companies

        2020-03-30/一季度近半企业利润预减13家逆势翻番.txt
        2019-08-12/二季度房贷增速继续回落楼市降温痕迹隐现.txt
        2016-11-07/有色金属10只个股年报净利润预计翻番.txt
        2018-06-11/5月份CPI同比涨幅与上月持平推高通胀因素正逐步退却.txt
        2013-09-27/统计局三因素致8月工业企业利润环比增速加快.txt
        2014-12-12/两融交易延续火爆融资关注能源股.txt
        2012-07-21/央企前6月净利润同比下降16%.txt
        2020-06-04/地摊概念再度爆发茂业商业等四连板多家公司提示风险.txt
        2013-01-16/文化传媒行业周报板块涨跌互现关注互联网发展带来的新媒体机会.txt
        2020-03-31/食品板块再度走强政策刺激消费关注受益板块反弹机会.txt
        2016-11-03/前三季度房地产市场去库存与高地价并存.txt
        2020-04-29/传媒娱乐股小幅走高华谊兄弟涨停拟引入阿里腾讯等9战投.txt
        2016-06-06/投资到位资金增速逐月回升固定资产投资后劲增强.txt
        2013-06-04/食品饮料两主线挖掘行业机遇.txt
        2019-04-03/游戏审批全面恢复网络游戏概念股掀起涨停潮.txt
        2020-03-10/半导体板块大幅拉升三星火灾叠加疫情影响存储器或再涨价.txt
        2014-07-12/新一轮电改风起云涌四领域尽享政策红利.txt
        2013-09-05/美联储褐皮书7月初至8月底间美国经济呈现温和增长.txt
        2012-10-29/广交会调查四季度仅两成企业订单增长.txt
        2013-12-19/央行调查显示超3成企业认为宏观经济偏冷.txt
    
    Cluster6:
        涨超/e/讯/证券时报/银行/发稿/拉升/涨停/
        Again, stock market reaching up limit, k=8 is too much...

        2019-10-21/银行股早盘全线飘红宁波银行涨超3%.txt
        2019-01-18/婴童概念股午后拉升奥飞娱乐涨超6%.txt
        2020-05-06/体育概念股盘中大幅拉升莱茵体育涨停.txt
        2020-06-12/9只银行股逆市上涨估值低位引关注.txt
        2019-11-13/黄金概念早盘集体大涨.txt
        2017-11-17/银行股走势活跃招商银行涨超3%.txt
        2020-05-12/新冠检测板块走强昌红科技等涨停.txt
        2019-10-22/网络安全概念股拉升任子行等涨停.txt
        2019-03-22/旅游板块大涨中青旅涨超9%.txt
        2018-07-20/银行股集体拉升成都银行涨停招商银行涨近5%.txt
        2020-05-13/特高压概念再度活跃保变电气涨停.txt
        2018-02-28/保险、银行、券商等金融板块持续走弱.txt
        2019-12-10/无线耳机概念股再度拉升惠威科技涨停.txt
        2019-11-08/汽车板块走强长安汽车涨超7%.txt
        2019-07-15/酿酒板块全线回调贵州茅台跌近3%.txt
        2019-03-06/互联金融概念持续走强大智慧12天10涨停.txt
        2019-09-24/白酒板块涨幅持续扩大金徽酒涨停.txt
        2019-04-04/工业大麻概念早盘拉升紫鑫药业涨停.txt
        2020-03-04/芯片股集体大跌兆易创新跌停.txt
        2019-10-09/科技股午后普遍回暖光刻胶指数涨超3%.txt
    
    Cluster7:
        的/基金/投资者/企业/市场/发行/资本/上市公司/
        Clearly about investment

        2017-11-24/投资者结构发生较大变化四类资金成主力.txt
        2012-08-23/证监会拟允许券商代销金融产品品种不受限.txt
        2016-02-23/两融余额8个月下跌幅度超六成.txt
        2015-01-26/股权众筹最具互联网基因今年有望出台监管方案.txt
        2016-02-22/深交所规范公司债券发行等业务.txt
        2016-12-28/证监会公司债券部多措并举开展投保工作.txt
        2015-10-12/常山药业(300255)拟定增6426万股募资8亿元.txt
        2018-04-16/对外投融资基金发展正当时为一带一路建设持续提供资金融通.txt
        2018-11-29/券商股盘中集体回落国海证券天风证券领跌.txt
        2019-07-09/中国通号等9家公司发行价出炉.txt
        2017-07-24/由点到面A股市场对外开放全面开花.txt
        2019-02-15/国办切实降低小微企业和三农综合融资成本.txt
        2020-03-02/券商板块震荡拉升关注政策呵护下券商投资价值.txt
        2015-12-23/央行银行间债券市场推出绿色金融债券.txt
        2019-04-30/国有资本授权经营体制改革将全面推广.txt
        2015-08-21/证金汇金概念股再添30家新丁医药行业受追捧.txt
        2019-05-24/专家市场化债转股核心在于推进合理定价机制.txt
        2014-12-31/军工集团整体上市可能是下一阶段改革重心.txt
        2018-11-09/A股又见活水慈善基金曲线入市可期.txt
        2016-10-18/深港通开通渐近内地资金追捧准美元资产.txt
    
    Cluster8:
        逾/涨/股份/5%/跌/科技/4%/走强/
        another group about stock market

        2019-10-29/养殖股走强益生股份触及涨停.txt
        2018-09-19/稀土永磁概念盘中走强鹏起科技涨逾6%.txt
        2019-09-09/铝业股快速走强常铝股份涨逾4%.txt
        2019-10-28/芯片股拉升长电科技涨逾7%.txt
        2017-12-13/次新股午后继续走强勘设股份等多股涨停.txt
        2019-02-14/白酒板块早盘稳步上扬山西汾酒、水井坊大涨4%.txt
        2017-07-13/稀土板块集体上攻北方稀土领涨.txt
        2019-08-21/磷概念拉升澄星股份涨停.txt
        2018-10-23/酿酒板块领跌洋河股份跌逾8%.txt
        2018-09-03/5G板块杀跌中兴通讯盘中跌逾8%.txt
        2019-08-29/深圳本地股卷土重来皇庭国际等多股涨停.txt
        2019-06-04/猪肉股杀跌天邦股份触及跌停.txt
        2018-10-29/股份回购板块爆发个股掀涨停潮.txt
        2019-03-13/军工板块逆市走强中国卫星等涨逾9%.txt
        2019-06-03/中兴通讯持续拉升涨逾8%大唐电信等5G概念股批量涨停.txt
        2019-05-21/华为海思板块回暖诚迈科技、兴森科技回封涨停.txt
        2019-10-23/数字货币概念持续活跃四方精创等涨停.txt
        2019-09-24/数字货币概念再度爆发金冠股份、亚联发展涨停.txt
        2019-06-14/稀土板块走低银河磁体等跌逾5%.txt
        2017-11-16/国足主帅表示中国将申办2030世界杯体育板块拉升.txt

6. Evaluation with k-8:
   Pro:
   as mentioned above, I believe the clustering is pretty successful, even dealing with one very specific form of articles- short financial articles 
   with cluster2 about macro-economic situstion report,
        cluster4 about currenct policy, 
        cluster7 about investment, 
    and cluster 1,3,5,8 about stock market.

   The biases could be simply because of the bias from the data, there is simply too many articles about stock market.
   if you lower down k from 8 to 5, some cluster would get mixed-up, thus the best k-score should be somewhat between 5-8
   but I really don't have time to check, as running the thing again is TOO SLOW.

   Con and needs to be improved:
   1. the database is biased with too many articles about stock market, thus biased
   2. topic-extraction algorithm needs to be improved
   3. manual dictionary needs to be added to better segment the word
   4. perhaps numbers could be eliminated to improve performancce

7. other notes:
   Don't forget to check the visualized groups, although they don't make much sense
   as the dimensionality is too high and everything got mixed together.




