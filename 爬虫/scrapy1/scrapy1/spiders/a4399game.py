import scrapy


class A4399gameSpider(scrapy.Spider):
    name = "4399game"       # 爬虫名字
    allowed_domains = ["4399.com"]      # 允许的域名
    start_urls = ["https://www.4399.com/flash/"]        # 起始页面url

    def parse(self, response, **kwargs):          # 该方法默认用来处理解析
        lst = response.xpath("//ul[@class='n-game cf']/li")
        for li in lst:
            name = li.xpath("./a/b/text()").extract_first()
            category = li.xpath("./em[1]/a/text()").extract_first()
            time = li.xpath("./em[2]/text()").extract_first()

            dic = {
                'name': name,
                "category": category,
                'time': time
            }
            # 需要用yield将数据传递给管道
            yield dic