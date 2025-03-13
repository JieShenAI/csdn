# 全国两地之间的距离与通勤时间

## 简介

调用百度地图的API：

* https://lbsyun.baidu.com/faq/api?title=webapi/guide/webservice-geocoding-base 地理编码
* https://lbsyun.baidu.com/faq/api?title=webapi/guide/webservice-lwrouteplanapi/dirve 驾车路线规划

全国333个市级单位的地址 `CityList_333.xlsx`



![image-20250313164511238](readme.assets/image-20250313164511238.png)

通过调用地理编码的API获取这些地址对应的经纬度,  `CityList_333_经纬度_百度API.xlsx`:

![image-20250313164655682](readme.assets/image-20250313164655682.png)

通过地址之间的两两组合，构建一个源地址与目标地址的

![image-20250313165022564](readme.assets/image-20250313165022564.png)

