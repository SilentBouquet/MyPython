import scrapy
from scrapy2.items import Scrapy2Item


class LotterySpider(scrapy.Spider):
    name = "lottery"
    allowed_domains = ["500.com"]
    start_urls = ["https://datachart.500.com/ssq/"]

    def parse(self, resp, **kwargs):
        print(resp.text)
        trs = resp.xpath("//tbody[@id='tdata']/tr")
        for tr in trs:
            if tr.xpath("./@class").extract_first() == "tdbck":
                continue
            red_ball = tr.xpath("./td[@class='chartBall01']/text()").extract()
            blue_ball = tr.xpath("./td[@class='chartBall02']/text()").extract_first()
            issue = tr.xpath("./td[1]/text()").extract_first().strip()

            lottery = Scrapy2Item()
            lottery['issue'] = issue
            lottery['red_ball'] = red_ball
            lottery['blue_ball'] = blue_ball

            yield lottery