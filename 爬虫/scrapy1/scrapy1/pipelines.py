# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html


# useful for handling different item types with a single interface
from itemadapter import ItemAdapter


# 管道默认是不生效的，需要去settings里面去开启管道
class Scrapy1Pipeline:
    def process_item(self, item, spider):       # 处理数据的专用方法，item是数据，spider是爬虫
        print(item)
        return item