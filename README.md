# DC_power_prediction
智慧中国杯数据应用大赛 国能日新光伏功率预测
DC 光伏赛
相关数据可在DataCastle官网下载

赛题：
  通过主办方给定的十个电站历史发电数据及气象数据，建立模型预测未来一段时间的光伏发电量；
 
成绩：8/636
  
1、数据预处理

    删除了一些异常发电的数据；

2、特征工程

    高阶气象特征
    交互特征
    差分特征
    光伏领域相关特征
    比率特征
    点击率特征
  
3、

    模型主要使用lightgbm
